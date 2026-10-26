import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sgemm_cuda as sgemm_native
from cuda_kernel_lab import sgemm

from benchmarks.common import positive_int


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One deterministic SGEMM launch for a profiler"
    )
    parser.add_argument(
        "--implementation",
        choices=sgemm.IMPLEMENTATIONS,
        default=sgemm.DEFAULT_IMPLEMENTATION,
    )
    parser.add_argument("--m", type=positive_int, default=4096)
    parser.add_argument("--n", type=positive_int, default=4096)
    parser.add_argument("--k", type=positive_int, default=4096)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()

    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA device is required")

    spec = sgemm.BACKENDS[args.implementation]
    if spec.tile_multiple is not None:
        tm, tn, tk = spec.tile_multiple
        if args.m % tm or args.n % tn or args.k % tk:
            parser.error(
                f"{args.implementation} requires M/N/K multiples "
                f"{spec.tile_multiple}"
            )

    generator = torch.Generator(device="cuda").manual_seed(2026)
    a = torch.randn((args.m, args.k), generator=generator, device="cuda")
    b = torch.randn((args.k, args.n), generator=generator, device="cuda")
    for _ in range(args.warmup):
        sgemm_native.forward(a, b, args.implementation)
    torch.cuda.synchronize()

    torch.cuda.nvtx.range_push(args.implementation)
    sgemm_native.forward(a, b, args.implementation)
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()

