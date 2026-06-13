# strix.rs NPU: MEMTILE multi-core flash attention (the whole_array-class lever).
# Solves the three walls that capped the simple N-pipelines version:
#   - >8 cores: direct shim→core exhausts shim DMA channels. Memtiles aggregate.
#   - KV replication: host replicated KV NQT× (DMA + BD-limit wall at bucket-512).
#     Here KV is loaded ONCE and BROADCAST from the memtile to all cores.
#   - re-streaming: at 1 tile/core (NC=NQT), each core does ONE query tile and
#     streams the (broadcast) KV blocks once — no re-stream needed.
# Pattern (per whole_array): Q shim→memtile→SPLIT to cores; KV shim→memtile→
# FORWARD+broadcast (multiple .cons()); O cores→memtile→JOIN→shim.
# Host input layout: [ Q (NQT*MT*D, tile order) | KV (NBLK blocks, ONE copy) ].
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
    assert MQ % MT == 0 and L % LB == 0
    TPH = MQ // MT
    NQT = NH * TPH
    NBLK = L // LB
    assert NQT == NC, "this design uses exactly 1 query tile per core (NC == NQT)"
    QREG = NH * MQ * D
    T = QREG + NBLK * 2 * LB * D  # KV stored ONCE (not replicated)

    in_ty = np.ndarray[(T,), np.dtype[bfloat16]]
    o_all_ty = np.ndarray[(NH * MQ * D,), np.dtype[bfloat16]]
    q_ty = np.ndarray[(MT * D,), np.dtype[bfloat16]]
    o_ty = np.ndarray[(MT * D,), np.dtype[bfloat16]]
    kv_ty = np.ndarray[(2 * LB * D,), np.dtype[bfloat16]]
    mf_ty = np.ndarray[(MT,), np.dtype[np.float32]]
    of_ty = np.ndarray[(MT * D,), np.dtype[np.float32]]

    k_block = Kernel("attn_block", "attention.o", [q_ty, kv_ty, mf_ty, mf_ty, of_ty, np.int32, np.int32])
    k_fin = Kernel("attn_finalize", "attention.o", [of_ty, mf_ty, mf_ty, o_ty])

    # --- Q: shim→memtile, then SPLIT one tile to each core ---
    # depth=1 on the compute-tile (L1) side: double-buffering would overflow the
    # 64KB tile (q 8KB + kv 16KB + o 8KB + o_buf f32 16KB already ≈ 48KB at depth-1).
    q_l3l2 = ObjectFifo(np.ndarray[(NC * MT * D,), np.dtype[bfloat16]], name="qL3L2", depth=2)
    q_l2l1 = q_l3l2.cons().split(
        [c * MT * D for c in range(NC)],
        obj_types=[q_ty] * NC,
        names=[f"qL2L1_{c}" for c in range(NC)],
        depths=[2] * NC,
    )

    # --- KV: shim→memtile, then FORWARD; broadcast to all cores via multiple cons ---
    kv_l3l2 = ObjectFifo(kv_ty, name="kvL3L2", depth=2)
    kv_l2l1 = kv_l3l2.cons().forward(obj_type=kv_ty, name="kvL2L1", depth=2)

    # --- O: each core → memtile → shim (JOIN) ---
    o_l2l3 = ObjectFifo(np.ndarray[(NC * MT * D,), np.dtype[bfloat16]], name="oL2L3", depth=2)
    o_l1l2 = o_l2l3.prod().join(
        [c * MT * D for c in range(NC)],
        obj_types=[o_ty] * NC,
        names=[f"oL1L2_{c}" for c in range(NC)],
        depths=[2] * NC,
    )

    workers = []
    for c in range(NC):
        m_buf = Buffer(type=mf_ty, initial_value=np.full((MT,), -3.0e38, dtype=np.float32), name=f"m{c}")
        l_buf = Buffer(type=mf_ty, initial_value=np.zeros((MT,), dtype=np.float32), name=f"l{c}")
        o_buf = Buffer(type=of_ty, initial_value=np.zeros((MT * D,), dtype=np.float32), name=f"ob{c}")
        pt_const = c % TPH  # this core owns global tile c → position-tile c%TPH

        def core_fn(q_in, kv_in, o_out, mb, lb, ob, kblk, kfin, pt_const=pt_const):
            eq = q_in.acquire(1)
            for kb in range_(NBLK):  # KV blocks (broadcast from memtile)
                ek = kv_in.acquire(1)
                # pt_const is a Python int constant (this core's fixed tile); kb is
                # the runtime loop index → index_cast to i32 for the causal arg.
                kblk(eq, ek, mb, lb, ob, pt_const, index_cast(kb, to=Ty.i32()))
                kv_in.release(1)
            eo = o_out.acquire(1)
            kfin(ob, lb, mb, eo)
            o_out.release(1)
            q_in.release(1)

        workers.append(
            Worker(
                core_fn,
                [q_l2l1[c].cons(), kv_l2l1.cons(), o_l1l2[c].prod(), m_buf, l_buf, o_buf, k_block, k_fin],
                stack_size=0xA00,
            )
        )

    rt = Runtime()
    with rt.sequence(in_ty, o_all_ty) as (IN, O):
        for w in workers:
            rt.start(w)
        rt.fill(q_l3l2.prod(), IN, tap=TensorAccessPattern((T,), 0, [NC * MT, D], [D, 1]))
        rt.fill(kv_l3l2.prod(), IN, tap=TensorAccessPattern((T,), QREG, [NBLK * 2 * LB, D], [D, 1]))
        rt.drain(o_l2l3.cons(), O, tap=TensorAccessPattern((NH * MQ * D,), 0, [NC * MT, D], [D, 1]), wait=True)

    return Program(NPU2(), rt).resolve_program()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("-M", type=int, default=32)
    ap.add_argument("--mq", type=int, default=0)
    ap.add_argument("-K", type=int, default=128)
    ap.add_argument("-N", type=int, default=128)
    ap.add_argument("--lb", type=int, default=32)
    ap.add_argument("--nh", type=int, default=1)
    ap.add_argument("--nc", type=int, default=4)
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
