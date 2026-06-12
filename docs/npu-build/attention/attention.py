# strix.rs NPU stage-3 prototype: STREAMING single-core flash attention.
# Q resident (loaded once); K/V stream block-by-block (one K‖V block at a time)
# so the L axis is not resident — lifts the 64KB-tile ceiling. Running (m,l,o)
# state lives in persistent core-local Buffers across the block loop.
# Emits MLIR for aiecc. Shapes via -M/-K(=L)/-N(=D); block size via --lb.
import argparse
import numpy as np
from ml_dtypes import bfloat16

from aie.iron import Buffer, Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2
from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessPattern


def attention(M, L, D, LB):
    assert L % LB == 0, "block size LB must divide L"
    NBLK = L // LB
    # Host input layout (reordered so each K/V block is contiguous):
    #   [ Q (M*D) | block0 (K0‖V0, 2*LB*D) | block1 | ... ]
    T = M * D + 2 * L * D

    in_ty = np.ndarray[(T,), np.dtype[bfloat16]]
    o_ty = np.ndarray[(M * D,), np.dtype[bfloat16]]
    q_ty = np.ndarray[(M * D,), np.dtype[bfloat16]]
    kv_ty = np.ndarray[(2 * LB * D,), np.dtype[bfloat16]]
    mf_ty = np.ndarray[(M,), np.dtype[np.float32]]
    of_ty = np.ndarray[(M * D,), np.dtype[np.float32]]

    k_block = Kernel("attn_block", "attention.o", [q_ty, kv_ty, mf_ty, mf_ty, of_ty])
    k_fin = Kernel("attn_finalize", "attention.o", [of_ty, mf_ty, o_ty])

    of_q = ObjectFifo(q_ty, name="q", depth=1)
    of_kv = ObjectFifo(kv_ty, name="kv", depth=2)  # double-buffer the K/V stream
    of_o = ObjectFifo(o_ty, name="o", depth=1)

    m_buf = Buffer(type=mf_ty, initial_value=np.full((M,), -3.0e38, dtype=np.float32))
    l_buf = Buffer(type=mf_ty, initial_value=np.zeros((M,), dtype=np.float32))
    o_buf = Buffer(type=of_ty, initial_value=np.zeros((M * D,), dtype=np.float32))

    def core_fn(q_in, kv_in, o_out, mb, lb, ob, kb, kfin):
        eq = q_in.acquire(1)  # Q resident, reused across all KV blocks
        # Uniform streaming loop over KV blocks (matches the matmul K-reduction
        # dataflow). attn_block self-detects the first block from the m sentinel.
        for _ in (range_(NBLK) if NBLK > 1 else range(NBLK)):
            ek = kv_in.acquire(1)
            kb(eq, ek, mb, lb, ob)
            kv_in.release(1)
        # normalize + emit
        eo = o_out.acquire(1)
        kfin(ob, lb, eo)
        o_out.release(1)
        q_in.release(1)

    worker = Worker(
        core_fn,
        [of_q.cons(), of_kv.cons(), of_o.prod(), m_buf, l_buf, o_buf, k_block, k_fin],
        stack_size=0xA00,
    )

    rt = Runtime()
    with rt.sequence(in_ty, o_ty) as (IN, O):
        rt.start(worker)
        # 2D (rows × D) access patterns: a flat size-(M*D) transfer lowers to a
        # single repeat_count = M*D-1, which exceeds the DMA BD [0:255] limit.
        # Splitting into M rows of D contiguous keeps the outer repeat small.
        rt.fill(of_q.prod(), IN, tap=TensorAccessPattern((T,), 0, [M, D], [D, 1]))
        # ONE fill delivers all NBLK K/V objects (outer dim walks blocks), which
        # the fifo auto-segments into NBLK consumer acquires — the proven matmul
        # streaming idiom. Separate per-block fills collide on the same fifo slot.
        rt.fill(
            of_kv.prod(),
            IN,
            tap=TensorAccessPattern((T,), M * D, [NBLK, 2 * LB, D], [2 * LB * D, D, 1]),
        )
        rt.drain(of_o.cons(), O, wait=True)

    return Program(NPU2(), rt).resolve_program()


if __name__ == "__main__":
    # Reuses the matmul Makefile arg-set: M->queries, K->L (keys/values), N->D.
    ap = argparse.ArgumentParser()
    ap.add_argument("-M", type=int, default=64)
    ap.add_argument("-K", type=int, default=64)
    ap.add_argument("-N", type=int, default=64)
    ap.add_argument("--lb", type=int, default=32)
    ap.add_argument("--dev", type=str, default="npu2")
    ap.add_argument("--dtype_in", type=str, default="bf16")
    ap.add_argument("--dtype_out", type=str, default="bf16")
    ap.add_argument("--b-col-maj", type=int, default=0)
    ap.add_argument("--emulate-bf16-mmul-with-bfp16", type=str, default="false")
    ap.add_argument("--trace_size", type=int, default=0)
    ap.add_argument("--generate-taps", action="store_true")
    a = ap.parse_args()
    print(attention(a.M, a.K, a.N, a.lb))
