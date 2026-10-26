import pytest
import torch
import torch.nn.functional as F

cuda_ext = pytest.importorskip("flash_attn_cuda")

IMPLEMENTATIONS = (
    ("simt_fp32_pairwise_reg_scores", 2e-5, 2e-5),
    ("simt_fp32_pairwise_smem_scores", 2e-5, 2e-5),
    ("simt_fp32_pairwise_compact_smem_scores", 2e-5, 2e-5),
    ("tensor_core_tf32_wmma", 5e-3, 5e-3),
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("implementation,atol,rtol", IMPLEMENTATIONS)
@pytest.mark.parametrize("seq_len", (1, 17, 63, 64, 65, 127, 128, 257))
def test_forward_matches_sdpa(
    implementation: str,
    atol: float,
    rtol: float,
    seq_len: int,
) -> None:
    torch.manual_seed(2026)
    q = torch.randn(2, 3, seq_len, 64, device="cuda", dtype=torch.float32)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    expected = F.scaled_dot_product_attention(q, k, v)
    actual = cuda_ext.forward(q, k, v, implementation)

    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_uses_current_stream() -> None:
    q = torch.randn(1, 1, 65, 64, device="cuda")
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    stream = torch.cuda.Stream()

    with torch.cuda.stream(stream):
        actual = cuda_ext.forward(q, k, v)
        expected = F.scaled_dot_product_attention(q, k, v)

    stream.synchronize()
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_rejects_wrong_head_dim() -> None:
    q = torch.randn(1, 1, 64, 32, device="cuda")
    with pytest.raises(RuntimeError, match="head_dim=64"):
        cuda_ext.forward(q, q, q)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_rejects_mismatched_shapes() -> None:
    q = torch.randn(1, 1, 64, 64, device="cuda")
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    with pytest.raises(RuntimeError, match="identical shapes"):
        cuda_ext.forward(q, k[..., :32], v)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_rejects_mismatched_dtype() -> None:
    q = torch.randn(1, 1, 64, 64, device="cuda")
    with pytest.raises(RuntimeError, match="float32"):
        cuda_ext.forward(q, q.to(torch.float16), q)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_rejects_unknown_implementation() -> None:
    q = torch.randn(1, 1, 64, 64, device="cuda")
    with pytest.raises(RuntimeError, match="Unknown implementation"):
        cuda_ext.forward(q, q, q, "does_not_exist")
