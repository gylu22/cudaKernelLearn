#include <cuda_runtime.h>
#include <stdio.h>
#include <float.h>

#include "flash_attn/launchers.h"

// 宏定义：用于 128-bit 向量化读写，极大提升内存带宽利用率
#define FLOAT4(v) (reinterpret_cast<float4*>(&(v))[0])
#define CONST_FLOAT4(v) (reinterpret_cast<const float4*>(&(v))[0])

// LD = 68 (64 + 4)。
// 作用1：打破 D=64 带来的 Shared Memory Bank Conflict。
// 作用2：68 * 4 Bytes = 272 Bytes，272 % 16 == 0，完美保证每一行的起始地址满足 float4 的 16 字节对齐要求。
#define LD 68 

template<const int BR = 64, const int BC = 64, const int D = 64>
__global__ void simt_fp32_pairwise_reg_scores_kernel(
    const float* Q, const float* K, const float* V, float* O,
    int seq_len, int num_heads) 
{
    // -----------------------------------------------------------
    // 1. 全局索引与偏移量计算 (假设 Layout 为 [batch, num_heads, seq_len, head_dim])
    // -----------------------------------------------------------
    int batch_idx = blockIdx.z;
    int head_idx = blockIdx.y;
    int q_block_idx = blockIdx.x; // 当前处理的 Q 分块索引

    int batch_stride = num_heads * seq_len * D;
    int head_stride = seq_len * D;
    
    // Q, K, V, O 的基础偏移
    int base_offset = batch_idx * batch_stride + head_idx * head_stride;
    int q_offset = base_offset + q_block_idx * BR * D;

    // -----------------------------------------------------------
    // 2. 动态 Shared Memory 划分
    // -----------------------------------------------------------
    extern __shared__ float smem[];
    // 内存池布局: Q (BR x LD) -> K (BC x LD) -> V (BC x LD)
    float* smem_Q = smem;                              // 大小: 64 * 68 = 4352 个 float
    float* smem_K = smem + BR * LD;                    // 大小: 64 * 68 = 4352 个 float
    float* smem_V = smem + BR * LD + BC * LD;          // 大小: 64 * 68 = 4352 个 float

    int tid = threadIdx.x; // 0 ~ 127 (128 threads / 4 warps)
    
    // -----------------------------------------------------------
    // 3. 协同加载 Q 分块到 Shared Memory
    // -----------------------------------------------------------
    // 128 个线程协同搬运 64x64 的 Q。转为 float4 即 1024 个 float4。
    // 每个线程搬运 8 个 float4 (8 * 128 = 1024)
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int idx = i * 128 + tid; 
        int row = idx / 16;      // Q 的行 (0 ~ 63)
        int col_f4 = idx % 16;   // Q 的列组 (0 ~ 15)
        
        if (q_block_idx * BR + row < seq_len) {
            FLOAT4(smem_Q[row * LD + col_f4 * 4]) = CONST_FLOAT4(Q[q_offset + row * D + col_f4 * 4]);
        } else {
            FLOAT4(smem_Q[row * LD + col_f4 * 4]) = make_float4(0.f, 0.f, 0.f, 0.f);
        }
    }
    __syncthreads();

    // -----------------------------------------------------------
    // 4. GEMM 线程映射 (16x2 Grid) 与 Q 矩阵的寄存器常驻
    // -----------------------------------------------------------
    int warp_id = tid / 32;     // 0 ~ 3
    int lane_id = tid % 32;     // 0 ~ 31
    int row_id = lane_id / 2;   // 0 ~ 15 (当前线程负责 Warp 内的第几行)
    int col_id = lane_id % 2;   // 0 ~ 1  (孪生线程标识：0 负责偶数 K 列，1 负责奇数 K 列)

    int local_q_row = warp_id * 16 + row_id; // 在 SMEM_Q 中的行索引
    // int global_q_row = q_block_idx * BR + local_q_row; // 在全局 Q 中的行索引

    // 极致优化：将自己负责的这一行 Q 从 SMEM 读取到本地寄存器中！
    // 这样在内层循环时，无需再发生任何 SMEM 读取 Q 的操作。
    float q_reg[D]; 
    #pragma unroll
    for (int d = 0; d < D; d += 4) {
        FLOAT4(q_reg[d]) = FLOAT4(smem_Q[local_q_row * LD + d]);
    }

    // 寄存器状态初始化
    float o_acc[D] = {0.f}; // 贯穿整个内核的 O 累加器
    float row_m = -FLT_MAX; // 在线 Softmax 最大值
    float row_l = 0.f;      // 在线 Softmax 分母和
    float scale = 1.0f / sqrtf((float)D);

    // -----------------------------------------------------------
    // 5. 外层 K / V 循环
    // -----------------------------------------------------------
    int num_kv_blocks = (seq_len + BC - 1) / BC;
    for (int kv_idx = 0; kv_idx < num_kv_blocks; kv_idx++) {
        int kv_base = kv_idx * BC;
        
        // 5.1 协同加载 K 和 V 分块到 Shared Memory
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int idx = i * 128 + tid;
            int row = idx / 16;
            int col_f4 = idx % 16;
            int global_kv_row = kv_base + row;
            
            if (global_kv_row < seq_len) {
                FLOAT4(smem_K[row * LD + col_f4 * 4]) = CONST_FLOAT4(K[base_offset + global_kv_row * D + col_f4 * 4]);
                FLOAT4(smem_V[row * LD + col_f4 * 4]) = CONST_FLOAT4(V[base_offset + global_kv_row * D + col_f4 * 4]);
            } else {
                FLOAT4(smem_K[row * LD + col_f4 * 4]) = make_float4(0.f, 0.f, 0.f, 0.f);
                FLOAT4(smem_V[row * LD + col_f4 * 4]) = make_float4(0.f, 0.f, 0.f, 0.f);
            }
        }
        __syncthreads();

        // 5.2 阶段一：计算点积 (Score)
        float local_scores[32]; 
        float local_max = -FLT_MAX;
        
        #pragma unroll
        for(int s = 0; s < 32; s++) {
            int k_col = s * 2 + col_id; // col_id 为 0 的线程算 0,2,4..，为 1 的算 1,3,5..
            float sum = 0.f;
            
            // 纯寄存器与无 Bank Conflict SMEM 的极致 FMA
            #pragma unroll
            for (int d = 0; d < D; d++) {
                sum = fmaf(q_reg[d], smem_K[k_col * LD + d], sum);
            }
            
            float score = sum * scale;
            if (kv_base + k_col >= seq_len) {
                score = -FLT_MAX; // Mask
            }
            local_scores[s] = score;
            local_max = fmaxf(local_max, score);
        }

        // 5.3 孪生线程通信：交换并获取 64 个列的全局最大值
        // `__shfl_xor_sync` 参数 1 表示翻转 lane_id 的最低位 (0和1交换, 2和3交换)
        float twin_max = __shfl_xor_sync(0xffffffff, local_max, 1);
        float warp_max = fmaxf(local_max, twin_max);

        // 5.4 Online Softmax 状态更新
        float prev_m = row_m;
        row_m = fmaxf(row_m, warp_max);
        float decay = expf(prev_m - row_m);

        row_l *= decay;
        #pragma unroll
        for(int d = 0; d < D; d++) {
            o_acc[d] *= decay;
        }

        // 5.5 阶段二：计算指数并融合 V
        float local_l = 0.f;
        #pragma unroll
        for(int s = 0; s < 32; s++) {
            int k_col = s * 2 + col_id;
            float p = expf(local_scores[s] - row_m);
            local_l += p;
            
            #pragma unroll
            for(int d = 0; d < D; d++) {
                o_acc[d] = fmaf(p, smem_V[k_col * LD + d], o_acc[d]);
            }
        }

        // 5.6 孪生线程通信：累加分母和
        // 修复：需要同时累加本线程的 local_l 和 twin 的 local_l，才能得到完整的 softmax 分母
        float twin_l = __shfl_xor_sync(0xffffffff, local_l, 1);
        row_l += local_l + twin_l;

        // 等待所有线程用完当前的 SMEM K/V 块
        __syncthreads();
    }

    // -----------------------------------------------------------
    // 6. 收尾阶段：合并孪生线程的 o_acc 并协同写回全局内存
    // -----------------------------------------------------------
    
    // 孪生线程 0 积累了偶数行的 V，线程 1 积累了奇数行的 V，现在要把它们相加
    #pragma unroll
    for(int d = 0; d < D; d++) {
        float twin_o = __shfl_xor_sync(0xffffffff, o_acc[d], 1);
        o_acc[d] += twin_o; 
    }

    // 为了最大化全局内存带宽，复用 smem_Q 暂存归一化后的最终结果
    if (col_id == 0) { // 只需要一对孪生线程中的 1 个来写 SMEM
        #pragma unroll
        for(int d = 0; d < D; d += 4) {
            float4 res;
            res.x = o_acc[d+0] / row_l;
            res.y = o_acc[d+1] / row_l;
            res.z = o_acc[d+2] / row_l;
            res.w = o_acc[d+3] / row_l;
            FLOAT4(smem_Q[local_q_row * LD + d]) = res;
        }
    }
    __syncthreads();

    // 128 个线程再次组成 1024 个 Float4 矩阵，进行最完美的合并内存写回 (Coalesced Memory Access)
    #pragma unroll
    for (int i = 0; i < 8; i++) {
        int idx = i * 128 + tid;
        int row = idx / 16;
        int col_f4 = idx % 16;
        
        if (q_block_idx * BR + row < seq_len) {
            FLOAT4(O[q_offset + row * D + col_f4 * 4]) = FLOAT4(smem_Q[row * LD + col_f4 * 4]);
        }
    }
}


// Host 启动函数
cudaError_t flash_attn::launch_simt_fp32_pairwise_reg_scores_fwd(
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
    dim3 block(128); // 强制配置为 128 线程 (4 Warps)

    // 计算动态 Shared Memory 容量
    // Q (64x68) + K (64x68) + V (64x68) = 3 * 64 * 68 * sizeof(float)
    // 52224 Bytes (约 51 KB)
    int shared_mem_bytes = 3 * BR * LD * sizeof(float);

    // 申请提升动态 Shared Memory 限制 (默认通常限制在 48KB，提额后可达 96KB 或以上)
    cudaError_t status = cudaFuncSetAttribute(
        simt_fp32_pairwise_reg_scores_kernel<BR, BC, D>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_mem_bytes
    );
    if (status != cudaSuccess) {
        return status;
    }

    simt_fp32_pairwise_reg_scores_kernel<BR, BC, D>
        <<<grid, block, shared_mem_bytes, stream>>>(
            Q, K, V, O, seq_len, num_heads);

    return cudaGetLastError();
}
