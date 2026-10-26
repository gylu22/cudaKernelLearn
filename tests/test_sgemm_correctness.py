import pytest
import torch

sgemm_cuda = pytest.importorskip("sgemm_cuda")


GENERAL_IMPLEMENTATIONS = (
    "simt_fp32_00_naive",
    "simt_fp32_01_smem_tiled",
)

TILED_IMPLEMENTATIONS = (
    "simt_fp32_02_thread_tile_vec4",
    "simt_fp32_03_transposed_smem_vec4",
    "simt_fp32_04_double_buffer",
    "simt_fp32_05_padded_smem",
)


def test_rejects_cpu_inputs():
    a = torch.randn(8, 7)
    b = torch.randn(7, 9)
    with pytest.raises(RuntimeError, match="must be CUDA tensors"):
        sgemm_cuda.forward(a, b)


def test_rejects_rank_mismatch():
    a = torch.randn(2, 3, 4)
    b = torch.randn(4, 5)
    with pytest.raises(RuntimeError, match="rank-2 matrices"):
        sgemm_cuda.forward(a, b)


def make_inputs(m: int, n: int, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cuda").manual_seed(2026)
    a = torch.randn((m, k), device="cuda", generator=generator)
    b = torch.randn((k, n), device="cuda", generator=generator)
    return a, b


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("implementation", GENERAL_IMPLEMENTATIONS)
@pytest.mark.parametrize("shape", ((1, 1, 1), (17, 19, 13), (65, 33, 37)))
def test_general_backends_match_torch(implementation: str, shape: tuple[int, ...]):
    m, n, k = shape
    a, b = make_inputs(m, n, k)
    actual = sgemm_cuda.forward(a, b, implementation)
    expected = torch.matmul(a, b)
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("implementation", TILED_IMPLEMENTATIONS)
def test_tiled_backends_match_torch(implementation: str):
    a, b = make_inputs(128, 128, 16)
    actual = sgemm_cuda.forward(a, b, implementation)
    expected = torch.matmul(a, b)
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_uses_current_stream():
    a, b = make_inputs(65, 33, 37)
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        actual = sgemm_cuda.forward(a, b, "simt_fp32_01_smem_tiled")
        expected = torch.matmul(a, b)
    stream.synchronize()
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_rejects_incompatible_shapes():
    a = torch.randn(8, 7, device="cuda")
    b = torch.randn(6, 9, device="cuda")
    with pytest.raises(RuntimeError, match="Incompatible matrix shapes"):
        sgemm_cuda.forward(a, b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_rejects_noncontiguous_inputs():
    a = torch.randn(8, 7, device="cuda").t()
    b = torch.randn(8, 9, device="cuda")
    with pytest.raises(RuntimeError, match="contiguous"):
        sgemm_cuda.forward(a, b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_rejects_wrong_dtype():
    a = torch.randn(8, 8, device="cuda", dtype=torch.float16)
    b = torch.randn(8, 8, device="cuda", dtype=torch.float16)
    with pytest.raises(RuntimeError, match="float32"):
        sgemm_cuda.forward(a, b)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_rejects_unsupported_tiled_shape():
    a, b = make_inputs(127, 128, 8)
    with pytest.raises(RuntimeError, match="requires M and N divisible by 128"):
        sgemm_cuda.forward(a, b, "simt_fp32_02_thread_tile_vec4")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_rejects_unknown_implementation():
    a, b = make_inputs(8, 8, 8)
    with pytest.raises(RuntimeError, match="Unknown SGEMM implementation"):
        sgemm_cuda.forward(a, b, "does_not_exist")
