#pragma once

#include <cuda_runtime_api.h>

namespace sgemm {

// All launchers enqueue work on the supplied stream. The binding validates
// dtype, layout, device, and implementation-specific shape constraints.
using Launcher = cudaError_t (*)(
    const float* a,
    const float* b,
    float* c,
    int m,
    int n,
    int k,
    cudaStream_t stream);

cudaError_t launch_simt_fp32_00_naive(
    const float*, const float*, float*, int, int, int, cudaStream_t);
cudaError_t launch_simt_fp32_01_smem_tiled(
    const float*, const float*, float*, int, int, int, cudaStream_t);
cudaError_t launch_simt_fp32_02_thread_tile_vec4(
    const float*, const float*, float*, int, int, int, cudaStream_t);
cudaError_t launch_simt_fp32_03_transposed_smem_vec4(
    const float*, const float*, float*, int, int, int, cudaStream_t);
cudaError_t launch_simt_fp32_04_double_buffer(
    const float*, const float*, float*, int, int, int, cudaStream_t);
cudaError_t launch_simt_fp32_05_padded_smem(
    const float*, const float*, float*, int, int, int, cudaStream_t);

}  // namespace sgemm

