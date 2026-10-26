#include <torch/extension.h>

#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime_api.h>

#include <climits>
#include <string>

#include "sgemm/launchers.h"

namespace sgemm {
namespace {

struct Backend {
    Launcher launcher;
    const char* canonical_name;
    bool requires_128x128x8_tiles;
};

void check_inputs(const torch::Tensor& a, const torch::Tensor& b) {
    TORCH_CHECK(a.dim() == 2 && b.dim() == 2,
                "A and B must be rank-2 matrices");
    TORCH_CHECK(a.is_cuda() && b.is_cuda(),
                "A and B must be CUDA tensors");
    TORCH_CHECK(a.device() == b.device(),
                "A and B must be on the same CUDA device");
    TORCH_CHECK(a.scalar_type() == torch::kFloat32 &&
                    b.scalar_type() == torch::kFloat32,
                "SGEMM currently supports float32 only");
    TORCH_CHECK(a.is_contiguous() && b.is_contiguous(),
                "A and B must be contiguous row-major matrices");
    TORCH_CHECK(a.size(0) > 0 && a.size(1) > 0 && b.size(1) > 0,
                "M, N, and K must be positive");
    TORCH_CHECK(a.size(1) == b.size(0),
                "Incompatible matrix shapes: A=", a.sizes(),
                ", B=", b.sizes());
    TORCH_CHECK(a.size(0) <= INT_MAX && a.size(1) <= INT_MAX &&
                    b.size(1) <= INT_MAX,
                "Matrix dimensions exceed the launcher's 32-bit index range");
}

Backend find_backend(const std::string& implementation) {
    if (implementation == "simt_fp32_00_naive" ||
        implementation == "sgemm_naive_f32") {
        return {launch_simt_fp32_00_naive, "simt_fp32_00_naive", false};
    }
    if (implementation == "simt_fp32_01_smem_tiled" ||
        implementation == "sgemm_sliced_k_f32") {
        return {launch_simt_fp32_01_smem_tiled,
                "simt_fp32_01_smem_tiled", false};
    }
    if (implementation == "simt_fp32_02_thread_tile_vec4" ||
        implementation == "sgemm_t_8x8_sliced_k_f32x4") {
        return {launch_simt_fp32_02_thread_tile_vec4,
                "simt_fp32_02_thread_tile_vec4", true};
    }
    if (implementation == "simt_fp32_03_transposed_smem_vec4" ||
        implementation == "sgemm_t_8x8_sliced_k_bcf_f32x4") {
        return {launch_simt_fp32_03_transposed_smem_vec4,
                "simt_fp32_03_transposed_smem_vec4", true};
    }
    if (implementation == "simt_fp32_04_double_buffer" ||
        implementation == "sgemm_t_8x8_sliced_k_bcf_db_f32x4") {
        return {launch_simt_fp32_04_double_buffer,
                "simt_fp32_04_double_buffer", true};
    }
    if (implementation == "simt_fp32_05_padded_smem" ||
        implementation == "runSgemmResolveBankExtraCol") {
        return {launch_simt_fp32_05_padded_smem,
                "simt_fp32_05_padded_smem", true};
    }
    TORCH_CHECK(false, "Unknown SGEMM implementation '", implementation,
                "'. Available: simt_fp32_00_naive, "
                "simt_fp32_01_smem_tiled, simt_fp32_02_thread_tile_vec4, "
                "simt_fp32_03_transposed_smem_vec4, "
                "simt_fp32_04_double_buffer, simt_fp32_05_padded_smem");
    return {};
}

}  // namespace

torch::Tensor forward(const torch::Tensor& a,
                      const torch::Tensor& b,
                      const std::string& implementation) {
    check_inputs(a, b);
    const Backend backend = find_backend(implementation);

    const int m = static_cast<int>(a.size(0));
    const int k = static_cast<int>(a.size(1));
    const int n = static_cast<int>(b.size(1));
    if (backend.requires_128x128x8_tiles) {
        TORCH_CHECK(m % 128 == 0 && n % 128 == 0 && k % 8 == 0,
                    backend.canonical_name,
                    " currently requires M and N divisible by 128 and K "
                    "divisible by 8; got M=", m, ", N=", n, ", K=", k);
    }

    c10::cuda::CUDAGuard device_guard(a.device());
    // Zero initialization is intentional: the padded-SMEM teaching kernel
    // retains an alpha/beta epilogue and reads C even when beta is zero.
    auto out = torch::zeros({m, n}, a.options());
    const auto stream = c10::cuda::getCurrentCUDAStream(a.get_device());
    const cudaError_t status = backend.launcher(
        a.data_ptr<float>(), b.data_ptr<float>(), out.data_ptr<float>(),
        m, n, k, stream.stream());
    TORCH_CHECK(status == cudaSuccess, backend.canonical_name,
                " launch failed: ", cudaGetErrorString(status));
    return out;
}

}  // namespace sgemm

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &sgemm::forward,
          pybind11::arg("a"), pybind11::arg("b"),
          pybind11::arg("implementation") = "simt_fp32_01_smem_tiled",
          "SGEMM forward with an explicitly selected backend");
}

