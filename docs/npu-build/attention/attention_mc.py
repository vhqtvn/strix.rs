# strix.rs NPU multi-week I4: MULTI-CORE flash attention.
# Attention is embarrassingly parallel over query tiles (each query row is
# independent — no cross-core reduction, unlike matmul's K-reduction). So we run
# NC independent pipelines: core c processes a contiguous chunk of the NQT query
# tiles, with its own Q/KV/out fifos + (m,l,o) Buffers. Host input layout is
# IDENTICAL to single-core ([Q | KV replicated]); core c just reads its slice.
# Uses the matvec kernel (attention.o: attn_block/attn_finalize) to de-risk the
# multi-core dataflow before folding in the mmul kernel.
import argparse
import numpy as np
from ml_dtypes import bfloat16

from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2
from aie.iron.controlflow import range_
from aie.extras.dialects.arith import index_cast
from aie.extras import types as Ty
from aie.helpers.taplib import TensorAccessPattern


def attention(MT, MQ, L, D, LB, NH, NC):
    assert L % LB == 0 and MQ % MT == 0
    NBLK = L // LB
    TPH = MQ // MT
    NQT = NH * TPH
    assert NQT % NC == 0, "cores must divide total query tiles"
    TPC = NQT // NC  # query tiles per core
    T = NH * MQ * D + NQT * NBLK * 2 * LB * D
    QREG = NH * MQ * D  # offset where the (replicated) KV region begins

    in_ty = np.ndarray[(T,), np.dtype[bfloat16]]
    o_all_ty = np.ndarray[(NH * MQ * D,), np.dtype[bfloat16]]
    q_ty = np.ndarray[(MT * D,), np.dtype[bfloat16]]
    o_ty = np.ndarray[(MT * D,), np.dtype[bfloat16]]
    kv_ty = np.ndarray[(2 * LB * D,), np.dtype[bfloat16]]
    mf_ty = np.ndarray[(MT,), np.dtype[np.float32]]
    of_ty = np.ndarray[(MT * D,), np.dtype[np.float32]]

    k_block = Kernel("attn_block", "attention.o", [q_ty, kv_ty, mf_ty, mf_ty, of_ty, np.int32, np.int32])
    k_fin = Kernel("attn_finalize", "attention.o", [of_ty, mf_ty, mf_ty, o_ty])

    rt = Runtime()
    workers = []
    fills = []  # (prod, tap) deferred to the runtime sequence
    for c in range(NC):
        of_q = ObjectFifo(q_ty, name=f"q{c}", depth=1)
        of_kv = ObjectFifo(kv_ty, name=f"kv{c}", depth=1)
        of_o = ObjectFifo(o_ty, name=f"o{c}", depth=1)
        m_buf = Buffer(type=mf_ty, initial_value=np.full((MT,), -3.0e38, dtype=np.float32), name=f"m{c}")
        l_buf = Buffer(type=mf_ty, initial_value=np.zeros((MT,), dtype=np.float32), name=f"l{c}")
        o_buf = Buffer(type=of_ty, initial_value=np.zeros((MT * D,), dtype=np.float32), name=f"o_buf{c}")
        base = c * TPC  # this core's first global query-tile index

        def core_fn(q_in, kv_in, o_out, mb, lb, ob, kblk, kfin, base=base):
            # range_ (runtime loop) → tiny ELF (unrolling TPC*NBLK overflowed at
            # bucket-512). pt = (base+lt)%TPH and kb index_cast to i32 for causal.
            for lt in range_(TPC):  # local tile index
                pt = index_cast((base + lt) % TPH, to=Ty.i32())
                eq = q_in.acquire(1)
                for kb in range_(NBLK):
                    ek = kv_in.acquire(1)
                    kblk(eq, ek, mb, lb, ob, pt, index_cast(kb, to=Ty.i32()))
                    kv_in.release(1)
                eo = o_out.acquire(1)
                kfin(ob, lb, mb, eo)
                o_out.release(1)
                q_in.release(1)

        workers.append(
            Worker(
                core_fn,
                [of_q.cons(), of_kv.cons(), of_o.prod(), m_buf, l_buf, o_buf, k_block, k_fin],
                stack_size=0xA00,
            )
        )
        # per-core fills/drains (slices of the shared Q / KV / O regions)
        q_off = base * MT * D
        kv_off = QREG + base * NBLK * 2 * LB * D
        o_off = base * MT * D
        fills.append((of_q.prod(), TensorAccessPattern((T,), q_off, [TPC, MT, D], [MT * D, D, 1])))
        fills.append((of_kv.prod(), TensorAccessPattern((T,), kv_off, [TPC * NBLK, 2 * LB, D], [2 * LB * D, D, 1])))
        fills.append(("DRAIN", of_o.cons(), TensorAccessPattern((NH * MQ * D,), o_off, [TPC, MT, D], [MT * D, D, 1])))

    with rt.sequence(in_ty, o_all_ty) as (IN, O):
        for w in workers:
            rt.start(w)
        for f in fills:
            if f[0] == "DRAIN":
                rt.drain(f[1], O, tap=f[2], wait=True)
            else:
                rt.fill(f[0], IN, tap=f[1])

    return Program(NPU2(), rt).resolve_program()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-M", type=int, default=32)
    ap.add_argument("--mq", type=int, default=0)
    ap.add_argument("-K", type=int, default=128)
    ap.add_argument("-N", type=int, default=128)
    ap.add_argument("--lb", type=int, default=32)
    ap.add_argument("--nh", type=int, default=1)
    ap.add_argument("--nc", type=int, default=2)  # AIE cores
    ap.add_argument("--kvdepth", type=int, default=1)
    ap.add_argument("--dev", type=str, default="npu2")
    ap.add_argument("--dtype_in", type=str, default="bf16")
    ap.add_argument("--dtype_out", type=str, default="bf16")
    ap.add_argument("--b-col-maj", type=int, default=0)
    ap.add_argument("--emulate-bf16-mmul-with-bfp16", type=str, default="false")
    ap.add_argument("--trace_size", type=int, default=0)
    ap.add_argument("--generate-taps", action="store_true")
    a = ap.parse_args()
    mq = a.mq if a.mq > 0 else a.M
    print(attention(a.M, mq, a.K, a.N, a.lb, a.nh, a.nc))
