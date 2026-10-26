import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import flash_attn_cuda as flash_attn_native
from cuda_kernel_lab import flash_attention

from benchmarks.common import Case, make_qkv, positive_int


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One deterministic FlashAttention launch for a profiler"
    )
    parser.add_argument(
        "--implementation",
        choices=flash_attention.IMPLEMENTATIONS,
        default=flash_attention.DEFAULT_IMPLEMENTATION,
    )
    parser.add_argument("--batch", type=positive_int, default=1)
    parser.add_argument("--heads", type=positive_int, default=1)
    parser.add_argument("--seq-len", type=positive_int, default=1024)
    parser.add_argument(
        "--warmup",
        type=int,
        default=10,
        help="Unprofiled launches used to initialize CUDA and stabilize clocks",
    )
    args = parser.parse_args()

    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA device is required")

    case = Case(args.batch, args.heads, args.seq_len)
    q, k, v = make_qkv(case)
    for _ in range(args.warmup):
        flash_attn_native.forward(q, k, v, args.implementation)
    torch.cuda.synchronize()

    torch.cuda.nvtx.range_push(args.implementation)
    flash_attn_native.forward(q, k, v, args.implementation)
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()

