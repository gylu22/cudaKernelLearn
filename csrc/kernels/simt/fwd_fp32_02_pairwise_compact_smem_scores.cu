#include <cuda_runtime.h>
#include <float.h>

#include "flash_attn/config.h"
#include "flash_attn/launchers.h"

#define FLOAT4(value) (reinterpret_cast<float4*>(&(value))[0])
#define CONST_FLOAT4(value) (reinterpret_cast<const float4*>(&(value))[0])

constexpr int kThreads = 128;
constexpr int kThreadsPerQuery = 2;
constexpr int kScoresPerThread = flash_attn::kKeyTile / kThreadsPerQuery;
constexpr int kScoreElements = kThreads * kScoresPerThread;
constexpr int kQkElements =
    flash_attn::kQueryTile * flash_attn::kPaddedHeadDim;
constexpr int kVElements =
    flash_attn::kKeyTile * flash_attn::kHeadDim;
constexpr int kSharedElements =
    kQkElements + kScoreElements + kVElements;
constexpr int kSharedBytes = kSharedElements * sizeof(float);

static_assert(kScoreElements == 4096);
static_assert(kSharedBytes == 49 * 1024);

template <int Br, int Bc, int D, int Ld>
__global__ void simt_fp32_pairwise_compact_smem_scores_kernel(
    const float* q,
    const float* k,
    const float* v,
    float* out,
    int seq_len,
    int num_heads) {
    const int tid = threadIdx.x;
    const int query_row = tid / kThreadsPerQuery;
    const int key_parity = tid % kThreadsPerQuery;

    const int head_stride = seq_len * D;
    const int batch_stride = num_heads * head_stride;
    const int base_offset =
        blockIdx.z * batch_stride + blockIdx.y * head_stride;
    const int query_offset =
        base_offset + blockIdx.x * Br * D;

    extern __shared__ float smem[];
    float* smem_qk = smem;
    float* smem_scores = smem_qk + Br * Ld;
    float* smem_v = smem_scores + kScoreElements;

    // Cooperative, vectorized Q load. Padding columns are intentionally unused.
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int index = i * kThreads + tid;
        const int row = index / 16;
        const int column = (index % 16) * 4;
        if (blockIdx.x * Br + row < seq_len) {
            FLOAT4(smem_qk[row * Ld + column]) =
                CONST_FLOAT4(q[query_offset + row * D + column]);
        } else {
            FLOAT4(smem_qk[row * Ld + column]) =
                make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }
    __syncthreads();

    // The adjacent thread pair intentionally keeps one complete Q row per
    // thread. A later stage will replace this with smaller sub-warp fragments.
    float q_registers[D];
    #pragma unroll
    for (int d = 0; d < D; d += 4) {
        FLOAT4(q_registers[d]) =
            FLOAT4(smem_qk[query_row * Ld + d]);
    }

    // smem_qk is reused for K below. All threads must finish reading Q first.
    __syncthreads();

    float output_accumulator[D] = {0.0f};
    float row_max = -FLT_MAX;
    float row_sum = 0.0f;
    const float softmax_scale = rsqrtf(static_cast<float>(D));

    const int num_key_tiles = (seq_len + Bc - 1) / Bc;
    for (int key_tile = 0; key_tile < num_key_tiles; ++key_tile) {
        const int key_offset = base_offset + key_tile * Bc * D;

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int index = i * kThreads + tid;
            const int row = index / 16;
            const int column = (index % 16) * 4;
            if (key_tile * Bc + row < seq_len) {
                FLOAT4(smem_qk[row * Ld + column]) =
                    CONST_FLOAT4(k[key_offset + row * D + column]);
                FLOAT4(smem_v[row * D + column]) =
                    CONST_FLOAT4(v[key_offset + row * D + column]);
            } else {
                FLOAT4(smem_qk[row * Ld + column]) =
                    make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                FLOAT4(smem_v[row * D + column]) =
                    make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
        __syncthreads();

        float local_max = -FLT_MAX;
        #pragma unroll
        for (int score_index = 0;
             score_index < kScoresPerThread;
             ++score_index) {
            const int key_row =
                score_index * kThreadsPerQuery + key_parity;
            float dot = 0.0f;
            #pragma unroll
            for (int d = 0; d < D; ++d) {
                dot = fmaf(
                    q_registers[d],
                    smem_qk[key_row * Ld + d],
                    dot);
            }

            float score = dot * softmax_scale;
            if (key_tile * Bc + key_row >= seq_len) {
                score = -FLT_MAX;
            }
            local_max = fmaxf(local_max, score);
            smem_scores[tid + score_index * kThreads] = score;
        }
        __syncthreads();

        const float peer_max =
            __shfl_xor_sync(0xffffffff, local_max, 1);
        const float tile_max = fmaxf(local_max, peer_max);
        const float previous_max = row_max;
        row_max = fmaxf(row_max, tile_max);
        const float correction = expf(previous_max - row_max);

        row_sum *= correction;
        #pragma unroll
        for (int d = 0; d < D; ++d) {
            output_accumulator[d] *= correction;
        }

        float local_sum = 0.0f;
        #pragma unroll
        for (int score_index = 0;
             score_index < kScoresPerThread;
             ++score_index) {
            const float probability = expf(
                smem_scores[tid + score_index * kThreads] - row_max);
            local_sum += probability;
            const int key_row =
                score_index * kThreadsPerQuery + key_parity;

            #pragma unroll
            for (int d = 0; d < D; ++d) {
                output_accumulator[d] = fmaf(
                    probability,
                    smem_v[key_row * D + d],
                    output_accumulator[d]);
            }
        }

        const float peer_sum =
            __shfl_xor_sync(0xffffffff, local_sum, 1);
        row_sum += local_sum + peer_sum;
        __syncthreads();
    }

    #pragma unroll
    for (int d = 0; d < D; ++d) {
        output_accumulator[d] +=
            __shfl_xor_sync(0xffffffff, output_accumulator[d], 1);
    }

    if (key_parity == 0) {
        const float inverse_sum = 1.0f / fmaxf(row_sum, 1.0e-12f);
        #pragma unroll
        for (int d = 0; d < D; d += 4) {
            const float4 result = make_float4(
                output_accumulator[d] * inverse_sum,
                output_accumulator[d + 1] * inverse_sum,
                output_accumulator[d + 2] * inverse_sum,
                output_accumulator[d + 3] * inverse_sum);
            FLOAT4(smem_qk[query_row * Ld + d]) = result;
        }
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int index = i * kThreads + tid;
        const int row = index / 16;
        const int column = (index % 16) * 4;
        if (blockIdx.x * Br + row < seq_len) {
            FLOAT4(out[query_offset + row * D + column]) =
                FLOAT4(smem_qk[row * Ld + column]);
        }
    }
}

cudaError_t flash_attn::launch_simt_fp32_pairwise_compact_smem_scores_fwd(
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

    cudaError_t status = cudaFuncSetAttribute(
        simt_fp32_pairwise_compact_smem_scores_kernel<
            kQueryTile, kKeyTile, kHeadDim, kPaddedHeadDim>,
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
    simt_fp32_pairwise_compact_smem_scores_kernel<
        kQueryTile, kKeyTile, kHeadDim, kPaddedHeadDim>
        <<<grid, block, kSharedBytes, stream>>>(
            q, k, v, out, seq_len, num_heads);
    return cudaGetLastError();
}
