from typing import TYPE_CHECKING

from .registry import BackendSpec

if TYPE_CHECKING:
    import torch


BACKENDS = {
    spec.name: spec
    for spec in (
        BackendSpec("simt_fp32_00_naive", "simt", "float32"),
        BackendSpec("simt_fp32_01_smem_tiled", "simt", "float32"),
        BackendSpec(
            "simt_fp32_02_thread_tile_vec4", "simt", "float32",
            tile_multiple=(128, 128, 8),
        ),
        BackendSpec(
            "simt_fp32_03_transposed_smem_vec4", "simt", "float32",
            tile_multiple=(128, 128, 8),
        ),
        BackendSpec(
            "simt_fp32_04_double_buffer", "simt", "float32",
            tile_multiple=(128, 128, 8),
        ),
        BackendSpec(
            "simt_fp32_05_padded_smem", "simt", "float32",
            tile_multiple=(128, 128, 8),
        ),
    )
}
IMPLEMENTATIONS = tuple(BACKENDS)
DEFAULT_IMPLEMENTATION = "simt_fp32_01_smem_tiled"


def forward(
    a: "torch.Tensor",
    b: "torch.Tensor",
    implementation: str = DEFAULT_IMPLEMENTATION,
) -> "torch.Tensor":
    import torch  # noqa: F401 - load PyTorch libraries before the extension
    import sgemm_cuda

    return sgemm_cuda.forward(a, b, implementation)
