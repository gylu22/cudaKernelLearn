import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import flash_attn_cuda as flash_attn_native
from cuda_kernel_lab import flash_attention

from benchmarks.common import (
    Case,
    benchmark_cuda,
    make_qkv,
    make_result_document,
    positive_int,
    write_result_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark selectable FlashAttention forward backends"
    )
    parser.add_argument("--batch", type=positive_int, default=2)
    parser.add_argument("--heads", type=positive_int, default=8)
    parser.add_argument(
        "--seq-len", type=positive_int, nargs="+", default=[64, 256, 1024]
    )
    parser.add_argument(
        "--implementation",
        choices=flash_attention.IMPLEMENTATIONS,
        nargs="+",
        default=[flash_attention.DEFAULT_IMPLEMENTATION],
    )
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=positive_int, default=100)
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

    document = make_result_document(
        "flash_attention_forward",
        config={
            "batch": args.batch,
            "heads": args.heads,
            "seq_len": args.seq_len,
            "implementations": args.implementation,
            "warmup": args.warmup,
            "iterations": args.iterations,
            "repeats": args.repeats,
            "timing_scope": (
                "CUDA-event interval around repeated native extension calls; "
                "output allocation remains inside each call"
            ),
        },
    )

    print(f"GPU: {document['environment']['gpu']}")
    print("Latency is the median of repeated CUDA-event samples (ms/call).")

    for seq_len in args.seq_len:
        case = Case(args.batch, args.heads, seq_len)
        q, k, v = make_qkv(case)
        reference = F.scaled_dot_product_attention(q, k, v)
        print(f"\nshape={case.shape}")

        for implementation in args.implementation:
            spec = flash_attention.BACKENDS[implementation]
            output = flash_attn_native.forward(q, k, v, implementation)
            torch.testing.assert_close(
                output, reference, atol=spec.atol, rtol=spec.rtol
            )
            error = (output - reference).abs()
            timing = benchmark_cuda(
                lambda implementation=implementation: flash_attn_native.forward(
                    q, k, v, implementation
                ),
                warmup=args.warmup,
                iterations=args.iterations,
                repeats=args.repeats,
            )
            print(
                f"  {implementation:42s} {timing.median_ms:9.4f} ms"
                f"  range=[{timing.minimum_ms:.4f}, {timing.maximum_ms:.4f}]"
                f"  max_abs_error={error.max().item():.3e}"
            )
            document["results"].append(
                {
                    "implementation": implementation,
                    "shape": list(case.shape),
                    "status": "ok",
                    "timing": timing.as_dict(),
                    "max_abs_error": error.max().item(),
                    "mean_abs_error": error.mean().item(),
                }
            )

        timing = benchmark_cuda(
            lambda: F.scaled_dot_product_attention(q, k, v),
            warmup=args.warmup,
            iterations=args.iterations,
            repeats=args.repeats,
        )
        print(
            f"  {'torch_sdpa':42s} {timing.median_ms:9.4f} ms"
            f"  range=[{timing.minimum_ms:.4f}, {timing.maximum_ms:.4f}]"
        )
        document["results"].append(
            {
                "implementation": "torch_sdpa",
                "shape": list(case.shape),
                "status": "reference",
                "timing": timing.as_dict(),
            }
        )

    if args.output_json:
        output = write_result_json(args.output_json, document)
        print(f"\nJSON: {output}")


if __name__ == "__main__":
    main()
