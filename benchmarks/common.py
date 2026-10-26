"""Shared utilities for reproducible CUDA benchmarks."""

from __future__ import annotations

import json
import platform
import statistics
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import torch


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
RESULT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class Case:
    batch: int = 2
    heads: int = 8
    seq_len: int = 1024
    head_dim: int = 64

    @property
    def shape(self) -> tuple[int, int, int, int]:
        return self.batch, self.heads, self.seq_len, self.head_dim


@dataclass(frozen=True)
class TimingStats:
    """Per-call latency derived from repeated CUDA-event measurements."""

    median_ms: float
    minimum_ms: float
    maximum_ms: float
    samples_ms: tuple[float, ...]
    iterations_per_sample: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise ValueError(f"expected a positive integer, got {value}")
    return parsed


def make_qkv(
    case: Case,
    *,
    seed: int = 2026,
    dtype: torch.dtype = torch.float32,
    device: str = "cuda",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device).manual_seed(seed)
    q = torch.randn(case.shape, generator=generator, device=device, dtype=dtype)
    k = torch.randn(case.shape, generator=generator, device=device, dtype=dtype)
    v = torch.randn(case.shape, generator=generator, device=device, dtype=dtype)
    return q, k, v


def benchmark_cuda(
    fn: Callable[[], Any],
    *,
    warmup: int,
    iterations: int,
    repeats: int,
) -> TimingStats:
    """Measure repeated groups with CUDA events and report their median."""

    if warmup < 0:
        raise ValueError("warmup must be non-negative")
    if iterations <= 0 or repeats <= 0:
        raise ValueError("iterations and repeats must be positive")

    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) / iterations)

    return TimingStats(
        median_ms=statistics.median(samples),
        minimum_ms=min(samples),
        maximum_ms=max(samples),
        samples_ms=tuple(samples),
        iterations_per_sample=iterations,
    )


def _git_output(*args: str) -> str | None:
    try:
        result = subprocess.run(
            ("git", *args),
            cwd=REPOSITORY_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip()


def environment_metadata() -> dict[str, Any]:
    device = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(device)
    git_status = _git_output("status", "--porcelain")
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "gpu": torch.cuda.get_device_name(device),
        "compute_capability": f"{major}.{minor}",
        "git_commit": _git_output("rev-parse", "HEAD"),
        "git_dirty": bool(git_status) if git_status is not None else None,
    }


def make_result_document(
    benchmark: str,
    *,
    config: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "benchmark": benchmark,
        "environment": environment_metadata(),
        "config": config,
        "results": [],
    }


def write_result_json(path: str | Path, document: dict[str, Any]) -> Path:
    output = Path(path)
    if not output.is_absolute():
        output = REPOSITORY_ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(document, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return output

