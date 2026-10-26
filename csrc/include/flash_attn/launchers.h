#pragma once

#include <cuda_runtime_api.h>

namespace flash_attn {

// All launchers are asynchronous and enqueue work on the supplied stream.
// The binding layer owns validation; kernels may assume contiguous BHND FP32
// tensors with D=64 and matching Q/K/V shapes.
cudaError_t launch_simt_fp32_pairwise_reg_scores_fwd(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    cudaStream_t stream);

cudaError_t launch_simt_fp32_pairwise_smem_scores_fwd(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    cudaStream_t stream);

cudaError_t launch_simt_fp32_pairwise_compact_smem_scores_fwd(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    cudaStream_t stream);

cudaError_t launch_tensor_core_tf32_wmma_fwd(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    cudaStream_t stream);

}  // namespace flash_attn
