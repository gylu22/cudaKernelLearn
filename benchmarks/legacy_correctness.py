"""Manual correctness sweep kept for interactive investigation.

The automated test suite remains the source of truth. This script is useful
when inspecting error distributions for selected FlashAttention backends.
"""

import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import flash_attn_cuda as flash_attn_native
from cuda_kernel_lab import flash_attention

from benchmarks.common import Case, make_qkv, positive_int


def check_forward(implementation: str, case: Case) -> None:
    spec = flash_attention.BACKENDS[implementation]
    q, k, v = make_qkv(case)
    expected = F.scaled_dot_product_attention(q, k, v)
    actual = flash_attn_native.forward(q, k, v, implementation)
    error = (actual - expected).abs()

    torch.testing.assert_close(
        actual, expected, atol=spec.atol, rtol=spec.rtol
    )
    print(
        f"{implementation:42s} PASS"
        f"  max_abs={error.max().item():.3e}"
        f"  mean_abs={error.mean().item():.3e}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare active FlashAttention backends with PyTorch SDPA"
    )
    parser.add_argument(
        "--implementation",
        choices=("all", *flash_attention.IMPLEMENTATIONS),
        default="all",
    )
    parser.add_argument("--batch", type=positive_int, default=2)
    parser.add_argument("--heads", type=positive_int, default=8)
    parser.add_argument(
        "--seq-len", type=positive_int, nargs="+", default=[64, 65, 1024]
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA device is required")

    implementations = (
        flash_attention.IMPLEMENTATIONS
        if args.implementation == "all"
        else (args.implementation,)
    )
    for seq_len in args.seq_len:
        case = Case(args.batch, args.heads, seq_len)
        print(f"\nshape={case.shape}")
        for implementation in implementations:
            check_forward(implementation, case)


if __name__ == "__main__":
    main()

