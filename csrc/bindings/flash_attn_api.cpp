#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>

#include <climits>
#include <string>

#include "flash_attn/config.h"
#include "flash_attn/launchers.h"

namespace flash_attn {
namespace {

void check_input(const torch::Tensor& q,
                 const torch::Tensor& k,
                 const torch::Tensor& v) {
    TORCH_CHECK(q.dim() == 4, "Q must have shape [batch, heads, seq_len, head_dim]");
    TORCH_CHECK(k.dim() == 4 && v.dim() == 4,
                "K and V must have shape [batch, heads, seq_len, head_dim]");
    TORCH_CHECK(q.sizes() == k.sizes() && q.sizes() == v.sizes(),
                "Q, K, and V must have identical shapes; got Q=", q.sizes(),
                ", K=", k.sizes(), ", V=", v.sizes());
    TORCH_CHECK(q.is_cuda() && k.is_cuda() && v.is_cuda(),
                "Q, K, and V must be CUDA tensors");
    TORCH_CHECK(q.device() == k.device() && q.device() == v.device(),
                "Q, K, and V must be on the same CUDA device");
    TORCH_CHECK(q.scalar_type() == torch::kFloat32 &&
                k.scalar_type() == torch::kFloat32 &&
                v.scalar_type() == torch::kFloat32,
                "The SIMT kernels currently support float32 only");
    TORCH_CHECK(q.is_contiguous() && k.is_contiguous() && v.is_contiguous(),
                "Q, K, and V must be contiguous in BHND layout");
    TORCH_CHECK(q.size(0) > 0 && q.size(1) > 0 && q.size(2) > 0,
                "batch, heads, and seq_len must be positive");
    TORCH_CHECK(q.size(3) == kHeadDim,
                "The current SIMT kernels require head_dim=", kHeadDim,
                "; got ", q.size(3));
    TORCH_CHECK(q.size(0) <= INT_MAX && q.size(1) <= INT_MAX &&
                q.size(2) <= INT_MAX,
                "Tensor dimensions exceed the launcher's 32-bit index range");
}

using Launcher = cudaError_t (*)(
    const float*, const float*, const float*, float*,
    int, int, int, int, cudaStream_t);

torch::Tensor run_forward(const torch::Tensor& q,
                          const torch::Tensor& k,
                          const torch::Tensor& v,
                          Launcher launcher,
                          const char* implementation) {
    check_input(q, k, v);
    c10::cuda::CUDAGuard device_guard(q.device());

    auto out = torch::empty_like(q);
    const auto stream = c10::cuda::getCurrentCUDAStream(q.get_device());
    const cudaError_t status = launcher(
        q.data_ptr<float>(),
        k.data_ptr<float>(),
        v.data_ptr<float>(),
        out.data_ptr<float>(),
        static_cast<int>(q.size(0)),
        static_cast<int>(q.size(1)),
        static_cast<int>(q.size(2)),
        static_cast<int>(q.size(3)),
        stream.stream());

    TORCH_CHECK(status == cudaSuccess, implementation,
                " launch failed: ", cudaGetErrorString(status));
    return out;
}

}  // namespace

torch::Tensor simt_fp32_pairwise_reg_scores_forward(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v) {
    return run_forward(
        q, k, v, launch_simt_fp32_pairwise_reg_scores_fwd,
        "simt_fp32_pairwise_reg_scores");
}

torch::Tensor simt_fp32_pairwise_smem_scores_forward(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v) {
    return run_forward(
        q, k, v, launch_simt_fp32_pairwise_smem_scores_fwd,
        "simt_fp32_pairwise_smem_scores");
}

torch::Tensor simt_fp32_pairwise_compact_smem_scores_forward(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v) {
    return run_forward(
        q, k, v, launch_simt_fp32_pairwise_compact_smem_scores_fwd,
        "simt_fp32_pairwise_compact_smem_scores");
}

torch::Tensor tensor_core_tf32_wmma_forward(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v) {
    return run_forward(
        q, k, v, launch_tensor_core_tf32_wmma_fwd,
        "tensor_core_tf32_wmma");
}

torch::Tensor forward(const torch::Tensor& q,
                      const torch::Tensor& k,
                      const torch::Tensor& v,
                      const std::string& implementation) {
    if (implementation == "simt_fp32_pairwise_reg_scores" ||
        implementation == "simt_register_tiled") {
        return simt_fp32_pairwise_reg_scores_forward(q, k, v);
    }
    if (implementation == "simt_fp32_pairwise_smem_scores" ||
        implementation == "simt_shared_scores") {
        return simt_fp32_pairwise_smem_scores_forward(q, k, v);
    }
    if (implementation == "simt_fp32_pairwise_compact_smem_scores") {
        return simt_fp32_pairwise_compact_smem_scores_forward(q, k, v);
    }
    if (implementation == "tensor_core_tf32_wmma") {
        return tensor_core_tf32_wmma_forward(q, k, v);
    }
    TORCH_CHECK(false, "Unknown implementation '", implementation,
                "'. Available: simt_fp32_pairwise_reg_scores, "
                "simt_fp32_pairwise_smem_scores, "
                "simt_fp32_pairwise_compact_smem_scores, "
                "tensor_core_tf32_wmma");
    return {};
}

}  // namespace flash_attn

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &flash_attn::forward,
          pybind11::arg("q"), pybind11::arg("k"), pybind11::arg("v"),
          pybind11::arg("implementation") =
              "simt_fp32_pairwise_smem_scores",
          "FlashAttention forward with an explicitly selected backend");
    m.def("simt_fp32_pairwise_reg_scores_forward",
          &flash_attn::simt_fp32_pairwise_reg_scores_forward);
    m.def("simt_fp32_pairwise_smem_scores_forward",
          &flash_attn::simt_fp32_pairwise_smem_scores_forward);
    m.def("simt_fp32_pairwise_compact_smem_scores_forward",
          &flash_attn::simt_fp32_pairwise_compact_smem_scores_forward);
    m.def("tensor_core_tf32_wmma_forward",
          &flash_attn::tensor_core_tf32_wmma_forward);

    // Compatibility aliases for scripts written against earlier revisions.
    m.def("simt_register_tiled_forward",
          &flash_attn::simt_fp32_pairwise_reg_scores_forward);
    m.def("simt_shared_scores_forward",
          &flash_attn::simt_fp32_pairwise_smem_scores_forward);
    m.def("flash_attn_v2_gemm",
          &flash_attn::simt_fp32_pairwise_reg_scores_forward);
    m.def("flash_attn_v2_gemm_kv_reuse",
          &flash_attn::simt_fp32_pairwise_smem_scores_forward);
}
