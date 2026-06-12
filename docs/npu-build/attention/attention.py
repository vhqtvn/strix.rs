# strix.rs NPU stage-2 prototype: single-core fused attention design.
# scores=Q·K^T → softmax → ·V, all in one AIE core (intermediates local).
# Emits MLIR for aiecc. Shapes via -M/-L/-D (default 64x64x64, bf16).
import argparse
import numpy as np
from ml_dtypes import bfloat16

from aie.iron import Kernel, ObjectFifo, Program, Runtime, Worker
from aie.iron.device import NPU2


def attention(M, L, D):
    # Q‖K‖V packed into one input (a tile has only 2 input DMA channels).
    in_ty = np.ndarray[(M * D + 2 * L * D,), np.dtype[bfloat16]]
    o_ty = np.ndarray[(M * D,), np.dtype[bfloat16]]

    att = Kernel("attention_bf16", "attention.o", [in_ty, o_ty])

    # depth=1 (single-buffer): the packed Q‖K‖V is large; double-buffering would
    # overflow the 64 KB tile local memory. One-shot forward needs no ping-pong.
    of_in = ObjectFifo(in_ty, name="qkv", depth=1)
    of_o = ObjectFifo(o_ty, name="o", depth=1)

    def core_fn(qkv, o, att):
        ein = qkv.acquire(1)
        eo = o.acquire(1)
        att(ein, eo)
        qkv.release(1)
        o.release(1)

    worker = Worker(
        core_fn,
        [of_in.cons(), of_o.prod(), att],
        stack_size=0xA00,
    )

    rt = Runtime()
    with rt.sequence(in_ty, o_ty) as (QKV, O):
        rt.start(worker)
        rt.fill(of_in.prod(), QKV)
        rt.drain(of_o.cons(), O, wait=True)

    return Program(NPU2(), rt).resolve_program()


if __name__ == "__main__":
    # Accept the matmul Makefile's arg-set so we can reuse makefile-common:
    #   M -> M (queries), K -> L (keys/values), N -> D (head dim). Rest ignored.
    ap = argparse.ArgumentParser()
    ap.add_argument("-M", type=int, default=64)
    ap.add_argument("-K", type=int, default=64)
    ap.add_argument("-N", type=int, default=64)
    ap.add_argument("--dev", type=str, default="npu2")
    ap.add_argument("--dtype_in", type=str, default="bf16")
    ap.add_argument("--dtype_out", type=str, default="bf16")
    ap.add_argument("--b-col-maj", type=int, default=0)
    ap.add_argument("--emulate-bf16-mmul-with-bfp16", type=str, default="false")
    ap.add_argument("--trace_size", type=int, default=0)
    ap.add_argument("--generate-taps", action="store_true")
    a = ap.parse_args()
    print(attention(a.M, a.K, a.N))
