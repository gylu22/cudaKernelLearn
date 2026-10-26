from typing import TYPE_CHECKING

from .registry import BackendSpec

if TYPE_CHECKING:
    import torch


BACKENDS = {
    spec.name: spec
    for spec in (
        BackendSpec(
            "simt_fp32_pairwise_reg_scores", "simt", "float32",
            atol=2e-5, rtol=2e-5,
        ),
        BackendSpec(
            "simt_fp32_pairwise_smem_scores", "simt", "float32",
            atol=2e-5, rtol=2e-5,
        ),
        BackendSpec(
            "simt_fp32_pairwise_compact_smem_scores", "simt", "float32",
            atol=2e-5, rtol=2e-5,
        ),
        BackendSpec(
            "tensor_core_tf32_wmma", "tensor_core", "float32",
            minimum_compute_capability=(8, 0), atol=5e-3, rtol=5e-3,
        ),
    )
}
IMPLEMENTATIONS = tuple(BACKENDS)
DEFAULT_IMPLEMENTATION = "simt_fp32_pairwise_smem_scores"


def forward(
    q: "torch.Tensor",
    k: "torch.Tensor",
    v: "torch.Tensor",
    implementation: str = DEFAULT_IMPLEMENTATION,
) -> "torch.Tensor":
    import torch  # noqa: F401 - load PyTorch libraries before the extension
    import flash_attn_cuda

    return flash_attn_cuda.forward(q, k, v, implementation)
