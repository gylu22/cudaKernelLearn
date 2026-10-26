#include <cuda_runtime.h>
#include <stdio.h>
#include <float.h>

#include "flash_attn/launchers.h"

// 宏定义：用于 128-bit 向量化读写，极大提升内存带宽利用率
#define FLOAT4(v) (reinterpret_cast<float4*>(&(v))[0])
#define CONST_FLOAT4(v) (reinterpret_cast<const float4*>(&(v))[0])

#define LD 68 



// ============================================================================
// v1 优化：消除 local_scores[32] 寄存器数组，Pass 2 重算 score 来融合 V
//   - 节省 128 bytes/thread 寄存器 → 降低 register pressure → 提升 occupancy
//   - 代价：QK 点积计算量翻倍 (2048 → 4096 FMA/tile/thread)
//   - 适用于 memory-bound 场景，额外的计算被 SMEM 带宽充裕所抵消
// ============================================================================
template<const int BR=64,const int BC=64,const int D=64>
__global__ void simt_fp32_pairwise_smem_scores_kernel(
    const float* Q, const float* K, const float* V, float* O,
    int seq_len, int num_heads){

    int head_offset = seq_len * D;
    int batch_offset = num_heads * head_offset;
    int base_offset = blockIdx.y * head_offset + blockIdx.z * batch_offset;
    int q_offset = blockIdx.x * BR * D + base_offset;



    
    extern __shared__ float smem[];
    float* smem_QK    = smem;                  // K tile:  BR × LD floats
    float* smem_Score = smem + BR * LD;        // Scores:  128 threads × 32 scores
    float* smem_V     = smem + 2 * BR * LD;    // V tile:  BR × LD floats
    int tid = threadIdx.x;

    // Q global -> SMEM
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int idx = i * 128 + tid;
        int row = idx / 16;
        int col = idx % 16 * 4;
        if (blockIdx.x * BR + row < seq_len) {
            FLOAT4(smem_QK[row * LD + col]) = CONST_FLOAT4(Q[q_offset + row * D + col]);
        } else {
            FLOAT4(smem_QK[row * LD + col]) = make_float4(0.f, 0.f, 0.f, 0.f);
        }
    }
    __syncthreads();

    // Q 常驻寄存器
    float q_reg[D];
    #pragma unroll
    for (int d = 0; d < D; d += 4) {
        FLOAT4(q_reg[d]) = FLOAT4(smem_QK[(tid / 2) * LD + d]);
    }

    // 状态初始化
    float o_acc[D] = {0.f};
    float row_m = -FLT_MAX;
    float row_l = 0.f;
    float scale = 1.0f / sqrtf((float)D);

    int row_id = tid / 2;
    int col_id = tid % 2;

    // KV 分块循环
    int num_kv_blocks = (seq_len + BC - 1) / BC;
    for (int kv_idx = 0; kv_idx < num_kv_blocks; kv_idx++) {
        // load K V tile to SMEM
        int kv_offset = kv_idx * BC * D + base_offset;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int idx = i * 128 + tid;
            int row = idx / 16;
            int col = idx % 16 * 4;
            if (kv_idx * BC + row < seq_len){
                FLOAT4(smem_QK[row * LD + col]) = CONST_FLOAT4(K[kv_offset + row * D + col]);
                FLOAT4(smem_V[row * LD + col]) = CONST_FLOAT4(V[kv_offset + row * D + col]);
            } else {
                FLOAT4(smem_QK[row * LD + col]) = make_float4(0.f, 0.f, 0.f, 0.f);
                FLOAT4(smem_V[row * LD + col]) = make_float4(0.f, 0.f, 0.f, 0.f);
            }
        }
        __syncthreads();

        // ============================================================
        // Step A: 计算所有 score，存入 SMEM（复用 smem_QK 的 K 区），同时 track local_max
        // K 数据在 score 读入寄存器后不再需要 → 用 smem_QK 暂存 score
        // 布局: thread tid 的 32 个 score 存于 smem_QK[tid + s * 128]
        //       (column-major, bank conflict free: bank = tid % 32)
        // ============================================================
        float local_max = -FLT_MAX;
        #pragma unroll
        for (int s = 0; s < 32; s++) {
            int k_col = s * 2 + col_id;
            float sum = 0.f;
            #pragma unroll
            for (int d = 0; d < D; d++) {
                sum = fmaf(q_reg[d], smem_QK[k_col * LD + d], sum);
            }
            float score = sum * scale;
            if (kv_idx * BC + k_col >= seq_len) {
                score = -FLT_MAX;
            }
            local_max = fmaxf(local_max, score);
            smem_Score[tid + s * 128] = score;  // 存入独立 score 缓冲区
        }
        __syncthreads();  // 确保所有 score 写入完毕，K 不再需要

        // Step B: warp shuffle → tile max → online softmax rescale
        float twin_max = __shfl_xor_sync(0xffffffff, local_max, 1);
        float tile_max = fmaxf(local_max, twin_max);
        float prev_m = row_m;
        row_m = fmaxf(row_m, tile_max);
        float decay = expf(prev_m - row_m);
        row_l *= decay;
        #pragma unroll
        for (int d = 0; d < D; d++) {
            o_acc[d] *= decay;
        }

        // Step C: 从 SMEM 读回 score → exp + V 融合 (与 v0 逻辑一致，但寄存器用量更少)
        float local_l = 0.f;
        #pragma unroll
        for (int s = 0; s < 32; s++) {
            float score = smem_Score[tid + s * 128];    // 从独立 score 缓冲区读回
            float p = expf(score - row_m);
            local_l += p;
            int kv_col = s * 2 + col_id;
            #pragma unroll
            for (int d = 0; d < D; d++) {
                o_acc[d] = fmaf(p, smem_V[kv_col * LD + d], o_acc[d]);
            }
        }

        float twin_l = __shfl_xor_sync(0xffffffff, local_l, 1);
        row_l += local_l + twin_l;
        __syncthreads();  // 确保所有线程读完 SMEM 才进行下一轮 K/V 加载
    }

    // 合并奇偶线程的 o_acc
    #pragma unroll
    for (int d = 0; d < D; d++) {
        float twin_o = __shfl_xor_sync(0xffffffff, o_acc[d], 1);
        o_acc[d] += twin_o;
    }

    // 归一化并写回 SMEM
    if (col_id == 0) {
        float inv_row_l = 1.0f / fmaxf(row_l, 1e-12f);
        for (int d = 0; d < D; d += 4) {
            float4 res;
            res.x = o_acc[d] * inv_row_l;
            res.y = o_acc[d + 1] * inv_row_l;
            res.z = o_acc[d + 2] * inv_row_l;
            res.w = o_acc[d + 3] * inv_row_l;
            FLOAT4(smem_QK[row_id * LD + d]) = res;
        }
    }
    __syncthreads();

    // 合并写回 global memory
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int idx = i * 128 + tid;
        int row = idx / 16;
        int col = idx % 16 * 4;
        if (blockIdx.x * BR + row < seq_len) {
            FLOAT4(O[q_offset + row * D + col]) = FLOAT4(smem_QK[row * LD + col]);
        }
    }
}




// Host 启动函数更新
cudaError_t flash_attn::launch_simt_fp32_pairwise_smem_scores_fwd(
    const float* Q, const float* K, const float* V, float* O,
    int bs, int num_heads, int seq_len, int head_dim, cudaStream_t stream)
{
    if (head_dim != 64) {
        return cudaErrorInvalidValue;
    }

    const int BR = 64;
    const int BC = 64;
    const int D = 64;
    
    int num_blocks = (seq_len + BR - 1) / BR;
    dim3 grid(num_blocks, num_heads, bs);
    dim3 block(128); 

    // v1: K + Score + V = 3 个 tile，各 BR*LD floats
    int shared_mem_bytes = 3 * BC * LD * sizeof(float); // 52,224 Bytes (约 51 KB)

    cudaError_t status = cudaFuncSetAttribute(
        simt_fp32_pairwise_smem_scores_kernel<BR, BC, D>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_mem_bytes
    );
    if (status != cudaSuccess) {
        return status;
    }

    simt_fp32_pairwise_smem_scores_kernel<BR, BC, D>
        <<<grid, block, shared_mem_bytes, stream>>>(
            Q, K, V, O, seq_len, num_heads);

    return cudaGetLastError();
}



