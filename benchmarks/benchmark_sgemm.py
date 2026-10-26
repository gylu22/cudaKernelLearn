import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import sgemm_cuda as sgemm_native
from cuda_kernel_lab import sgemm

from benchmarks.common import (
    benchmark_cuda,
    make_result_document,
    positive_int,
    write_result_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark selectable SGEMM backends")
    parser.add_argument("--m", type=positive_int, default=4096)
    parser.add_argument("--n", type=positive_int, default=4096)
    parser.add_argument("--k", type=positive_int, default=4096)
    parser.add_argument(
        "--implementation",
        choices=sgemm.IMPLEMENTATIONS,
        nargs="+",
        default=[sgemm.DEFAULT_IMPLEMENTATION],
    )
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=positive_int, default=50)
    parser.add_argument("--repeats", type=positive_int, default=5)
    parser.add_argument(
        "--output-json",
        help="Optional path for a machine-readable result document",
    )
    args = parser.parse_args()

    if args.warmup < 0:
        parser.error("--warmup must be non-negative")
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA device is required")

    torch.manual_seed(2026)
    torch.backends.cuda.matmul.allow_tf32 = False
    a = torch.randn((args.m, args.k), device="cuda")
    b = torch.randn((args.k, args.n), device="cuda")
    reference = torch.matmul(a, b)

    document = make_result_document(
        "sgemm",
        config={
            "m": args.m,
            "n": args.n,
            "k": args.k,
            "implementations": args.implementation,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "repeats": args.repeats,
            "allow_tf32_reference": False,
            "timing_scope": (
                "CUDA-event interval around repeated native extension calls; "
                "output allocation and device initialization remain inside each call"
            ),
        },
    )

    print(f"GPU: {document['environment']['gpu']}")
    print(f"shape: M={args.m}, N={args.n}, K={args.k}")
    print("Latency is the median of repeated CUDA-event samples (ms/call).")

    for implementation in args.implementation:
        spec = sgemm.BACKENDS[implementation]
        if spec.tile_multiple is not None:
            tm, tn, tk = spec.tile_multiple
            if args.m % tm or args.n % tn or args.k % tk:
                print(
                    f"  {implementation:42s} skipped"
                    f" (requires multiples {spec.tile_multiple})"
                )
                document["results"].append(
                    {
                        "implementation": implementation,
                        "shape": [args.m, args.n, args.k],
                        "status": "skipped",
                        "reason": f"requires multiples {spec.tile_multiple}",
                    }
                )
                continue

        output = sgemm_native.forward(a, b, implementation)
        torch.testing.assert_close(
            output, reference, atol=spec.atol, rtol=spec.rtol
        )
        error = (output - reference).abs()
        timing = benchmark_cuda(
            lambda implementation=implementation: sgemm_native.forward(
                a, b, implementation
            ),
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
        )
        tflops = 2.0 * args.m * args.n * args.k / (timing.median_ms * 1e9)
        print(
            f"  {implementation:42s} {timing.median_ms:9.4f} ms"
            f"  {tflops:8.2f} TFLOP/s"
            f"  range=[{timing.minimum_ms:.4f}, {timing.maximum_ms:.4f}]"
            f"  max_abs_error={error.max().item():.3e}"
        )
        document["results"].append(
            {
                "implementation": implementation,
                "shape": [args.m, args.n, args.k],
                "status": "ok",
                "timing": timing.as_dict(),
                "tflops": tflops,
                "max_abs_error": error.max().item(),
                "mean_abs_error": error.mean().item(),
            }
        )

    timing = benchmark_cuda(
        lambda: torch.matmul(a, b),
        warmup=args.warmup,
        iterations=args.iterations,
        repeats=args.repeats,
    )
    torch_tflops = 2.0 * args.m * args.n * args.k / (timing.median_ms * 1e9)
    print(
        f"  {'torch_matmul':42s} {timing.median_ms:9.4f} ms"
        f"  {torch_tflops:8.2f} TFLOP/s"
        f"  range=[{timing.minimum_ms:.4f}, {timing.maximum_ms:.4f}]"
    )
    document["results"].append(
        {
            "implementation": "torch_matmul",
            "shape": [args.m, args.n, args.k],
            "status": "reference",
            "timing": timing.as_dict(),
            "tflops": torch_tflops,
        }
    )

    if args.output_json:
        output = write_result_json(args.output_json, document)
        print(f"\nJSON: {output}")


if __name__ == "__main__":
    main()
