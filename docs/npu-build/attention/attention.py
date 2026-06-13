# strix.rs NPU stage-3 prototype: STREAMING + QUERY-TILED single-core flash attn.
# Queries are processed Mtile rows at a time (outer loop); for each query tile,
# K/V stream block-by-block (inner loop), one K‖V block resident. Running (m,l,o)
# state lives in persistent core-local Buffers, re-armed per query tile by
# attn_finalize. This bounds resident memory by Mtile (not total M) and by one
# K/V block (not L) — so real head_dims (D=128/256) and big seqs fit the 64KB tile.
# Shapes: -M = Mtile (rows/kernel call, == ATT_M macro), --mq = total queries,
#         -K = L (keys/values), -N = D (head dim), --lb = KV block size.
import argparse
import numpy as np
from ml_dtypes import bfloat16

from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2
from aie.iron.controlflow import range_
from aie.extras.dialects.arith import index_cast
from aie.extras import types as Ty  # 'T' is used below for the input-size
from aie.helpers.taplib import TensorAccessPattern


def attention(MT, MQ, L, D, LB, KVDEPTH=2, NH=1):
    assert L % LB == 0, "block size LB must divide L"
    assert MQ % MT == 0, "Mtile must divide queries-per-head"
    NBLK = L // LB
    TPH = MQ // MT       # position-tiles per query head
    NQT = NH * TPH       # total query tiles across all NH query heads (GQA group)
    # GQA: NH query heads share ONE streamed KV head. Query tiles are laid out
    # head-major (head0's TPH tiles, head1's, ...); each re-streams the shared KV.
    # The causal index passed per tile is the POSITION-tile within the head
    # (qti % TPH), so masking is per-position and head-independent.
    # Host input layout: [ Q (NH*MQ*D) | KV blocks REPLICATED per query tile ].
    # (AIE shim DMA has no stride-0 broadcast read → replicate KV NQT× in DDR;
    # a memtile broadcast would avoid the replication — perf TODO.)
    T = NH * MQ * D + NQT * NBLK * 2 * LB * D

    in_ty = np.ndarray[(T,), np.dtype[bfloat16]]
    o_all_ty = np.ndarray[(NH * MQ * D,), np.dtype[bfloat16]]
    q_ty = np.ndarray[(MT * D,), np.dtype[bfloat16]]
    o_ty = np.ndarray[(MT * D,), np.dtype[bfloat16]]
    kv_ty = np.ndarray[(2 * LB * D,), np.dtype[bfloat16]]
    mf_ty = np.ndarray[(MT,), np.dtype[np.float32]]
    of_ty = np.ndarray[(MT * D,), np.dtype[np.float32]]

    # qt, kb passed as scalar args from the dataflow loops (no persisted counter).
    k_block = Kernel("attn_block", "attention.o", [q_ty, kv_ty, mf_ty, mf_ty, of_ty, np.int32, np.int32])
    k_fin = Kernel("attn_finalize", "attention.o", [of_ty, mf_ty, mf_ty, o_ty])

    of_q = ObjectFifo(q_ty, name="q", depth=1)
    of_kv = ObjectFifo(kv_ty, name="kv", depth=KVDEPTH)  # double-buffer when it fits
    of_o = ObjectFifo(o_ty, name="o", depth=1)

    m_buf = Buffer(type=mf_ty, initial_value=np.full((MT,), -3.0e38, dtype=np.float32))
    l_buf = Buffer(type=mf_ty, initial_value=np.zeros((MT,), dtype=np.float32))
    o_buf = Buffer(type=of_ty, initial_value=np.zeros((MT * D,), dtype=np.float32))

    def core_fn(q_in, kv_in, o_out, mb, lb, ob, kblk, kfin):
        # Runtime range_ loops (NOT Python unroll): keeps the core ELF tiny so big
        # buckets/seqs fit program memory (unrolling NQT*NBLK calls overflowed the
        # ELF at bucket-512). The scf.for index is MLIR `index`; index_cast it to
        # i32 for the kernel's causal scalar args (pt = qti % TPH, kb).
        for qti in range_(NQT):  # query tiles (head-major across the GQA group)
            pt = index_cast(qti % TPH, to=Ty.i32())  # position-tile within head
            eq = q_in.acquire(1)
            for kb in range_(NBLK):  # KV blocks
                ek = kv_in.acquire(1)
                kblk(eq, ek, mb, lb, ob, pt, index_cast(kb, to=Ty.i32()))
                kv_in.release(1)
            eo = o_out.acquire(1)
            kfin(ob, lb, mb, eo)  # normalize + re-arm (m,l) for the next tile
            o_out.release(1)
            q_in.release(1)

    worker = Worker(
        core_fn,
        [of_q.cons(), of_kv.cons(), of_o.prod(), m_buf, l_buf, o_buf, k_block, k_fin],
        stack_size=0xA00,
    )

    rt = Runtime()
    with rt.sequence(in_ty, o_all_ty) as (IN, O):
        rt.start(worker)
        # Q: NQT objects of (Mtile × D), 2D inner pattern (keeps DMA repeat < 256).
        rt.fill(of_q.prod(), IN, tap=TensorAccessPattern((T,), 0, [NQT, MT, D], [MT * D, D, 1]))
        # K/V: NQT*NBLK objects, contiguous over the replicated region (no stride-0).
        rt.fill(
            of_kv.prod(),
            IN,
            tap=TensorAccessPattern((T,), NH * MQ * D, [NQT * NBLK, 2 * LB, D], [2 * LB * D, D, 1]),
        )
        # O: NQT output tiles (head-major).
        rt.drain(of_o.cons(), O, tap=TensorAccessPattern((NH * MQ * D,), 0, [NQT, MT, D], [MT * D, D, 1]), wait=True)

    return Program(NPU2(), rt).resolve_program()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-M", type=int, default=64)          # Mtile (rows / kernel call)
    ap.add_argument("--mq", type=int, default=0)         # total queries (0 → = Mtile)
    ap.add_argument("-K", type=int, default=64)          # L
    ap.add_argument("-N", type=int, default=64)          # D
    ap.add_argument("--lb", type=int, default=32)
    ap.add_argument("--nh", type=int, default=1)  # query heads sharing one KV head (GQA)
    ap.add_argument("--nc", type=int, default=1)  # ignored (single-core); see attention_mc.py
    ap.add_argument("--kvdepth", type=int, default=2)
    ap.add_argument("--dev", type=str, default="npu2")
    ap.add_argument("--dtype_in", type=str, default="bf16")
    ap.add_argument("--dtype_out", type=str, default="bf16")
    ap.add_argument("--b-col-maj", type=int, default=0)
    ap.add_argument("--emulate-bf16-mmul-with-bfp16", type=str, default="false")
    ap.add_argument("--trace_size", type=int, default=0)
    ap.add_argument("--generate-taps", action="store_true")
    a = ap.parse_args()
    mq = a.mq if a.mq > 0 else a.M
    print(attention(a.M, mq, a.K, a.N, a.lb, a.kvdepth, a.nh))
