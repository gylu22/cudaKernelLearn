#include <cuda_runtime.h>
#include <float.h>
#include <mma.h>

#include "flash_attn/config.h"
#include "flash_attn/launchers.h"

namespace wmma = nvcuda::wmma;

#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
#define CONST_FLOAT4(value) (reinterpret_cast<const float4*>(&(value))[0])

constexpr int kHeadDim = 64;
constexpr int kQueryTile = 16;
constexpr int kKeyTile = 64;
constexpr int kThreads = 128;
constexpr int kWarps = kThreads / 32;
constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 8;

constexpr int kQElements = kQueryTile * kHeadDim;
constexpr int kKElements = kKeyTile * kHeadDim;
constexpr int kVElements = kKeyTile * kHeadDim;
constexpr int kScoreElements = kQueryTile * kKeyTile;
constexpr int kOutputTileElements = kQueryTile * kHeadDim;
constexpr int kRowStateElements = 3 * kQueryTile;
constexpr int kSharedElements =
    kQElements + kKElements + kVElements +
    kScoreElements + kOutputTileElements + kRowStateElements;
constexpr int kSharedBytes = kSharedElements * sizeof(float);

static_assert(kWarps * kWmmaN == kKeyTile);
static_assert(kWarps * kWmmaN == kHeadDim);

__device__ __forceinline__ float to_tf32(float value) {
#if __CUDA_ARCH__ >= 800
    return wmma::__float_to_tf32(value);
#else
    return value;
#endif
}

__global__ void tensor_core_tf32_wmma_kernel(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    int seq_len,
    int num_heads) {
#if __CUDA_ARCH__ >= 800
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;

    const int head_stride = seq_len * kHeadDim;
    const int batch_stride = num_heads * head_stride;
    const int base_offset =
        blockIdx.z * batch_stride + blockIdx.y * head_stride;
    const int query_base = blockIdx.x * kQueryTile;

    extern __shared__ float smem[];
    float* smem_q = smem;
    float* smem_k = smem_q + kQElements;
    float* smem_v = smem_k + kKElements;
    float* smem_scores = smem_v + kVElements;
    float* smem_output_tile = smem_scores + kScoreElements;
    float* smem_row_max = smem_output_tile + kOutputTileElements;
    float* smem_row_sum = smem_row_max + kQueryTile;
    float* smem_row_correction = smem_row_sum + kQueryTile;

    // Q is reused across all K/V tiles. Convert the FP32 input to the TF32
    // representation explicitly before WMMA consumes it.
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int vector_index = i * kThreads + tid;
        const int element = vector_index * 4;
        const int row = element / kHeadDim;
        const int column = element % kHeadDim;

        float4 value = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (query_base + row < seq_len) {
            value = CONST_FLOAT4(
                q[base_offset + (query_base + row) * kHeadDim + column]);
        }
        value.x = to_tf32(value.x);
        value.y = to_tf32(value.y);
        value.z = to_tf32(value.z);
        value.w = to_tf32(value.w);
        FLOAT4(smem_q[element]) = value;
    }

    if (tid < kQueryTile) {
        smem_row_max[tid] = -FLT_MAX;
        smem_row_sum[tid] = 0.0f;
    }
    __syncthreads();

    // Each thread owns eight output elements across the entire K/V traversal.
    float output_accumulator[8] = {
        0.0f, 0.0f, 0.0f, 0.0f,
        0.0f, 0.0f, 0.0f, 0.0f};

    const int num_key_tiles = (seq_len + kKeyTile - 1) / kKeyTile;
    for (int key_tile = 0; key_tile < num_key_tiles; ++key_tile) {
        const int key_base = key_tile * kKeyTile;

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int vector_index = i * kThreads + tid;
            const int element = vector_index * 4;
            const int row = element / kHeadDim;
            const int column = element % kHeadDim;

            float4 key_value = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            float4 value_value = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            if (key_base + row < seq_len) {
                const int global_offset =
                    base_offset + (key_base + row) * kHeadDim + column;
                key_value = CONST_FLOAT4(k[global_offset]);
                value_value = CONST_FLOAT4(v[global_offset]);
            }

            key_value.x = to_tf32(key_value.x);
            key_value.y = to_tf32(key_value.y);
            key_value.z = to_tf32(key_value.z);
            key_value.w = to_tf32(key_value.w);
            value_value.x = to_tf32(value_value.x);
            value_value.y = to_tf32(value_value.y);
            value_value.z = to_tf32(value_value.z);
            value_value.w = to_tf32(value_value.w);
            FLOAT4(smem_k[element]) = key_value;
            FLOAT4(smem_v[element]) = value_value;
        }
        __syncthreads();

        // Four warps compute the four 16x16 column tiles of QK^T.
        wmma::fragment<
            wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float>
            score_accumulator;
        wmma::fill_fragment(score_accumulator, 0.0f);

        #pragma unroll
        for (int dimension_tile = 0;
             dimension_tile < kHeadDim;
             dimension_tile += kWmmaK) {
            wmma::fragment<
                wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK,
                wmma::precision::tf32, wmma::row_major>
                q_fragment;
            wmma::fragment<
                wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK,
                wmma::precision::tf32, wmma::col_major>
                k_fragment;

            wmma::load_matrix_sync(
                q_fragment,
                smem_q + dimension_tile,
                kHeadDim);
            wmma::load_matrix_sync(
                k_fragment,
                smem_k + warp_id * kWmmaN * kHeadDim + dimension_tile,
                kHeadDim);
            wmma::mma_sync(
                score_accumulator,
                q_fragment,
                k_fragment,
                score_accumulator);
        }

        const float scale = rsqrtf(static_cast<float>(kHeadDim));
        #pragma unroll
        for (int i = 0; i < score_accumulator.num_elements; ++i) {
            score_accumulator.x[i] *= scale;
        }
        wmma::store_matrix_sync(
            smem_scores + warp_id * kWmmaN,
            score_accumulator,
            kKeyTile,
            wmma::mem_row_major);
        __syncthreads();

        // One thread per query row performs the FP32 online softmax. The
        // probabilities are converted to TF32 only when written for the PV MMA.
        if (tid < kQueryTile) {
            const int row = tid;
            float tile_max = -FLT_MAX;
            #pragma unroll
            for (int column = 0; column < kKeyTile; ++column) {
                float score = smem_scores[row * kKeyTile + column];
                if (key_base + column >= seq_len) {
                    score = -FLT_MAX;
                }
                tile_max = fmaxf(tile_max, score);
            }

            const float previous_max = smem_row_max[row];
            const float new_max = fmaxf(previous_max, tile_max);
            const float correction = expf(previous_max - new_max);
            float tile_sum = 0.0f;

            #pragma unroll
            for (int column = 0; column < kKeyTile; ++column) {
                float probability = 0.0f;
                if (key_base + column < seq_len) {
                    probability = expf(
                        smem_scores[row * kKeyTile + column] - new_max);
                }
                tile_sum += probability;
                smem_scores[row * kKeyTile + column] =
                    to_tf32(probability);
            }

            smem_row_max[row] = new_max;
            smem_row_sum[row] =
                smem_row_sum[row] * correction + tile_sum;
            smem_row_correction[row] = correction;
        }
        __syncthreads();

        // Four warps compute four 16x16 column tiles of P*V.
        wmma::fragment<
            wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float>
            output_tile_accumulator;
        wmma::fill_fragment(output_tile_accumulator, 0.0f);

        #pragma unroll
        for (int key_dimension = 0;
             key_dimension < kKeyTile;
             key_dimension += kWmmaK) {
            wmma::fragment<
                wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK,
                wmma::precision::tf32, wmma::row_major>
                probability_fragment;
            wmma::fragment<
                wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK,
                wmma::precision::tf32, wmma::row_major>
                value_fragment;

            wmma::load_matrix_sync(
                probability_fragment,
                smem_scores + key_dimension,
                kKeyTile);
            wmma::load_matrix_sync(
                value_fragment,
                smem_v + key_dimension * kHeadDim + warp_id * kWmmaN,
                kHeadDim);
            wmma::mma_sync(
                output_tile_accumulator,
                probability_fragment,
                value_fragment,
                output_tile_accumulator);
        }

        wmma::store_matrix_sync(
            smem_output_tile + warp_id * kWmmaN,
            output_tile_accumulator,
            kHeadDim,
            wmma::mem_row_major);
        __syncthreads();

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int element = i * kThreads + tid;
            const int row = element / kHeadDim;
            output_accumulator[i] =
                output_accumulator[i] * smem_row_correction[row] +
                smem_output_tile[element];
        }
        __syncthreads();
    }

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int element = i * kThreads + tid;
        const int row = element / kHeadDim;
        const int column = element % kHeadDim;
        if (query_base + row < seq_len) {
            out[
                base_offset + (query_base + row) * kHeadDim + column] =
                output_accumulator[i] /
                fmaxf(smem_row_sum[row], 1.0e-12f);
        }
    }
#endif
}

cudaError_t flash_attn::launch_tensor_core_tf32_wmma_fwd(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    int batch_size,
    int num_heads,
    int seq_len,
    int head_dim,
    cudaStream_t stream) {
    if (head_dim != kHeadDim) {
        return cudaErrorInvalidValue;
    }

    int device = 0;
    cudaError_t status = cudaGetDevice(&device);
    if (status != cudaSuccess) {
        return status;
    }
    int major = 0;
    status = cudaDeviceGetAttribute(
        &major, cudaDevAttrComputeCapabilityMajor, device);
    if (status != cudaSuccess) {
        return status;
    }
    if (major < 8) {
        return cudaErrorNotSupported;
    }

    status = cudaFuncSetAttribute(
        tensor_core_tf32_wmma_kernel,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        kSharedBytes);
    if (status != cudaSuccess) {
        return status;
    }

    const dim3 grid(
        (seq_len + kQueryTile - 1) / kQueryTile,
        num_heads,
        batch_size);
    const dim3 block(kThreads);
    tensor_core_tf32_wmma_kernel
        <<<grid, block, kSharedBytes, stream>>>(
            q, k, v, out, seq_len, num_heads);
    return cudaGetLastError();
}
