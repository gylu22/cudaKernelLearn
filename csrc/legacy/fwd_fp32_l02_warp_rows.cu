#include <cuda_runtime.h>
#include <stdio.h>
#include <float.h>

#define FLOAT4(v) (reinterpret_cast<float4*>(&(v))[0])

// warp shuffle reduce max
__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        // 比较自己手中的值，和从 offset 处拿过来的值，保留较大的
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    // 0 号线程拿到全局最大值
    return val;
}

// warp shuffle reduce sum
__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}


template<const int BR=64, const int BC=64, int D=64>
__global__ void flash_attention_v2_fp32_forward_warp_kernel(
    float* Q,  float* K,  float* V, float* O, int seq_len, int num_heads){

        int batch_offset = blockIdx.z * num_heads * seq_len * D; // 每个 batch 的起始偏移
        int head_offset = batch_offset + blockIdx.y * seq_len * D; // 每个头的起始偏移
        int q_offset = head_offset + blockIdx.x * BR * D; // 当前 block 处理的 Q 的起始位置

        float* q_ptr = Q + q_offset; // 当前 block 处理的 Q 的起始位置
        float* k_ptr = K + head_offset; // 当前头的 K 的起始位置
        float* v_ptr = V + head_offset; // 当前头的 V 的起
        float* o_ptr = O + q_offset; // 当前 block 处理的 O 的起始位置

        extern __shared__ float smem[]; // 动态共享内存
        float* smem_Q = smem; // 存储当前 block 的 Q，大小为 BR * D
        float* smem_K = smem_Q + BR * D; // 存储当前 block 的 K，大小为 BC * D
        float* smem_V = smem_K + BC * D; // 存储当前 block 的 V，大小为 BC * D

        int tid = threadIdx.x;
        // int num_threads = blockDim.x;
        int warp_id = tid / 32; // 当前线程所在 warp 的 ID
        int lane_id = tid % 32; // 当前线程在 warp 中的 ID
        constexpr int NUM_WARPS = 256 / 32; // 8 个 warp

        int step = blockDim.x / (D / 4); // 256 / (64/4) = 16
        int row = tid / (D / 4); // 当前线程处理的 Q 的行索引
        int col = tid % (D / 4) * 4; // 当前线程处理的 Q 的列索引（以 float4 为单位）
        for (int q_i =0;q_i<BR;q_i+=step){
            int q_idx = (q_i + row) * D + col; // Q 的行索引
            if (q_i + row < BR){
                FLOAT4(smem_Q[q_idx]) = FLOAT4(q_ptr[q_idx]);
            }
        }
        __syncthreads(); 
        float o_acc[NUM_WARPS][2] = {0.f}; 
        float s_m[NUM_WARPS];      
        float s_l[NUM_WARPS] = {0.f};

        for (int i = 0; i < NUM_WARPS; i++) {
            s_m[i] = -FLT_MAX;
        }

        float scale = 1.0f / sqrtf((float)D);

        // 【修正 2+3】：去掉嵌套重复 kv_idx 循环和内层变量遮蔽，
        //             改为单层循环 O(N)；__syncthreads() 移到 kv_i 循环外部
        for (int kv_idx = 0; kv_idx < seq_len; kv_idx += BC) {

            // K/V: Global -> Shared（加上越界检查，防止最后一块 padding 导致 segfault）
            for (int kv_i = 0; kv_i < BC; kv_i += step){
                int kv_row = row + kv_idx + kv_i;
                if (kv_row < seq_len) {
                    FLOAT4(smem_K[(row + kv_i) * D + col]) = FLOAT4(k_ptr[kv_row * D + col]);
                    FLOAT4(smem_V[(row + kv_i) * D + col]) = FLOAT4(v_ptr[kv_row * D + col]);
                }
            }
            __syncthreads();

            // 核心计算部分：Warp 负责自己的 8 个 Query
            for (int q_step = 0; q_step < NUM_WARPS; q_step++) {
                int q_idx = q_step * NUM_WARPS + warp_id; // 0,1,2..7

                // 每个线程负责 D 维度中的 2 个元素
                float q[2];
                q[0] = smem_Q[q_idx * D + lane_id];
                q[1] = smem_Q[q_idx * D + lane_id + 32];

                // 内部循环 K, V 矩阵列 (BC)
                for (int k_idx = 0; k_idx < BC; k_idx++) {

                    // 1. 点积阶段 (Dot Product)
                    float k[2];
                    k[0] = smem_K[k_idx * D + lane_id];
                    k[1] = smem_K[k_idx * D + lane_id + 32];

                    float local_dot = q[0] * k[0] + q[1] * k[1];
                    float score = warp_reduce_sum(local_dot) * scale;

                    // 处理越界 Mask
                    if (kv_idx + k_idx >= seq_len) {
                        score = -FLT_MAX;
                    }

                    // 2. Online Softmax 阶段
                    float decay = 1.0f;
                    float p = 0.0f;
                    if (lane_id == 0) {
                        float prev_s_m = s_m[q_step];
                        s_m[q_step] = fmaxf(score, s_m[q_step]);
                        decay = expf(prev_s_m - s_m[q_step]);
                        p = expf(score - s_m[q_step]);
                        // 【修正 4】：使用 fmaf 提升数值精度
                        s_l[q_step] = fmaf(s_l[q_step], decay, p);
                    }

                    // 将 0 号线程的 decay 和 p 广播给 Warp 内的所有线程
                    decay = __shfl_sync(0xffffffff, decay, 0);
                    p     = __shfl_sync(0xffffffff, p, 0);

                    // 3. Value 融合阶段
                    float v[2];
                    v[0] = smem_V[k_idx * D + lane_id];
                    v[1] = smem_V[k_idx * D + lane_id + 32];

                    o_acc[q_step][0] = fmaf(o_acc[q_step][0], decay, p * v[0]);
                    o_acc[q_step][1] = fmaf(o_acc[q_step][1], decay, p * v[1]);
                }
            }
            // 等所有 Warp 用完当前 KV tile 再覆写共享内存
            __syncthreads();
        }

        // 结尾阶段：写回 Global Memory
        for (int q_step = 0; q_step < NUM_WARPS; q_step++) {
            int q_idx = q_step * NUM_WARPS + warp_id; 

            // 只有 0 号线程有准确的 s_l，广播给所有人做最后的归一化除法
            float final_s_l = __shfl_sync(0xffffffff, s_l[q_step], 0);

            // 如果 Query 没有越界，就写回
            if (blockIdx.x * BR + q_idx < seq_len) {
                o_ptr[q_idx * D + lane_id] = o_acc[q_step][0] / final_s_l;
                o_ptr[q_idx * D + lane_id + 32] = o_acc[q_step][1] / final_s_l;
            }
        }
}

// template<const int BR=64, const int BC=64, int D=64>
// __global__ void flash_attention_v2_fp32_forward_warp_kernel(
//     float* Q,  float* K,  float* V, float* O, int seq_len, int num_heads){

//     int batch_offset = blockIdx.z * num_heads * seq_len * D; 
//     int head_offset = batch_offset + blockIdx.y * seq_len * D; 
//     int q_offset = head_offset + blockIdx.x * BR * D; 

//     float* q_ptr = Q + q_offset; 
//     float* k_ptr = K + head_offset; 
//     float* v_ptr = V + head_offset; 
//     float* o_ptr = O + q_offset; 

//     extern __shared__ float smem[]; 
//     float* smem_Q = smem; 
//     float* smem_K = smem_Q + BR * D; 
//     float* smem_V = smem_K + BC * D; 

//     int tid = threadIdx.x;
//     int warp_id = tid / 32; // 当前线程所在 warp 的 ID (0~7)
//     int lane_id = tid % 32; // 当前线程在 warp 中的 ID (0~31)
//     constexpr int NUM_WARPS = 256 / 32; // 8 个 warp
    
//     // Q: Global -> Shared
//     int step = blockDim.x / (D / 4); // 256 / 16 = 16
//     int row = tid / (D / 4); 
//     int col = tid % (D / 4) * 4; 
//     for (int q_i = 0; q_i < BR; q_i += step){
//         int q_idx = (q_i + row) * D + col; 
//         if (q_i + row < BR){
//             FLOAT4(smem_Q[q_idx]) = FLOAT4(q_ptr[q_idx]);
//         }
//     }
    
//     // 【重要修正 1】：Q 加载完必须同步！
//     __syncthreads(); 

//     // 【重要修正 2】：修复 s_m 初始化 Bug，确保常量大小数组
//     float o_acc[NUM_WARPS][2] = {0.f}; 
//     float s_m[NUM_WARPS];      
//     float s_l[NUM_WARPS] = {0.f};
//     for (int i = 0; i < NUM_WARPS; i++) {
//         s_m[i] = -FLT_MAX;
//     }

//     float scale = 1.0f / sqrtf((float)D);

//     // 外层 KV 循环
//     for (int kv_idx = 0; kv_idx < seq_len; kv_idx += BC) {
        
//         // K/V: Global -> Shared
//         for (int kv_i = 0; kv_i < BC; kv_i += step){
//             int kv_row = row + kv_idx + kv_i; 
//             // 顺手加上越界检查，防止最后一块 padding 导致 segfault
//             if (kv_row < seq_len) {
//                 FLOAT4(smem_K[(row + kv_i) * D + col]) = FLOAT4(k_ptr[kv_row * D + col]);
//                 FLOAT4(smem_V[(row + kv_i) * D + col]) = FLOAT4(v_ptr[kv_row * D + col]);
//             }
//         }
//         __syncthreads(); 

//         // 核心计算部分：Warp 负责自己的 8 个 Query
//         for (int q_step = 0; q_step < NUM_WARPS; q_step++) {
//             // 当前 Warp 在这次循环中处理的 Query 在 Shared Memory 中的行号 (0~63)
//             // q_step 的步长是 1，由于 8 个 Warp 协同，实际上这里利用了你说的 "q_chunk+=num_warps" 思想
//             // 简单映射：Query_ID = q_step * 8 + warp_id 或者 warp_id * 8 + q_step
//             // 这里我们采用交错处理，Query_ID = q_step * NUM_WARPS + warp_id
//             int q_idx = q_step * NUM_WARPS + warp_id; 

//             // 每个线程负责 D 维度中的 2 个元素。
//             // 使用 lane_id 和 lane_id + 32，完美避免 Shared Memory Bank Conflict (无停顿访存)
//             float q[2];
//             q[0] = smem_Q[q_idx * D + lane_id];
//             q[1] = smem_Q[q_idx * D + lane_id + 32];

//             // 内部循环 K, V 矩阵列 (BC)
//             for (int k_idx = 0; k_idx < BC; k_idx++) {
                
//                 // 1. 点积阶段 (Dot Product)
//                 float k[2];
//                 k[0] = smem_K[k_idx * D + lane_id];
//                 k[1] = smem_K[k_idx * D + lane_id + 32];

//                 float local_dot = q[0] * k[0] + q[1] * k[1];
//                 float score = warp_reduce_sum(local_dot) * scale;

//                 // 处理越界 Mask (如果 K 的真实长度超过 seq_len)
//                 if (kv_idx + k_idx >= seq_len) {
//                     score = -FLT_MAX;
//                 }

//                 // 2. Online Softmax 阶段
//                 float decay = 1.0f;
//                 float p = 0.0f;
//                 if (lane_id == 0) {
//                     float prev_s_m = s_m[q_step];
//                     s_m[q_step] = fmaxf(score, s_m[q_step]);
//                     decay = expf(prev_s_m - s_m[q_step]);
//                     p = expf(score - s_m[q_step]);
//                     s_l[q_step] = fmaf(s_l[q_step], decay, p);
//                 }

//                 // 将 0 号线程的 decay 和 p 广播给 Warp 内的所有线程
//                 decay = __shfl_sync(0xffffffff, decay, 0);
//                 p     = __shfl_sync(0xffffffff, p, 0);

//                 // 3. Value 融合阶段
//                 float v[2];
//                 v[0] = smem_V[k_idx * D + lane_id];
//                 v[1] = smem_V[k_idx * D + lane_id + 32];

//                 // o_acc[q_step][0] = o_acc[q_step][0] * decay + p * v[0];
//                 // o_acc[q_step][1] = o_acc[q_step][1] * decay + p * v[1];
//                 o_acc[q_step][0] = fmaf(o_acc[q_step][0], decay, p * v[0]);
//                 o_acc[q_step][1] = fmaf(o_acc[q_step][1], decay, p * v[1]);
//             }
//         }
//         // 【重要同步】：等所有 Warp 把这块 Shared Memory 里的 K/V 用完了，再进入下一个 kv_idx 去覆写
//         __syncthreads(); 
//     }

//     // 结尾阶段：写回 Global Memory
//     for (int q_step = 0; q_step < NUM_WARPS; q_step++) {
//         int q_idx = q_step * NUM_WARPS + warp_id;
        
//         // 只有 0 号线程有准确的 s_l，广播给所有人做最后的归一化除法
//         float final_s_l = __shfl_sync(0xffffffff, s_l[q_step], 0);

//         // 如果 Query 没有越界，就写回
//         if (blockIdx.x * BR + q_idx < seq_len) {
//             o_ptr[q_idx * D + lane_id] = o_acc[q_step][0] / final_s_l;
//             o_ptr[q_idx * D + lane_id + 32] = o_acc[q_step][1] / final_s_l;
//         }
//     }
// }


void flash_attention_v2_fp32_warp_forward_host(
    float* Q,  float* K,  float* V, float* O, int bs, int num_heads, int seq_len, int head_dim){

    const int BR = 64; // 每个 block 处理的序列长度
    const int BC = 64; // 每个 block 处理的 K/V 列数
    int num_blocks = seq_len / BR; // 每个头的 block 数量

    dim3 grid(num_blocks, num_heads, bs);
    dim3 block(256); // 每个 block 256 个线程

    // 根据 head_dim 分发到正确的模板实例
    if (head_dim == 64) {
        const int D = 64;
        int shared_mem_bytes = (BC * D * 2 + BR * D) * sizeof(float);
        cudaFuncSetAttribute(
            flash_attention_v2_fp32_forward_warp_kernel<BR, BC, D>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shared_mem_bytes);
        flash_attention_v2_fp32_forward_warp_kernel<BR, BC, D><<<grid, block, shared_mem_bytes>>>(Q, K, V, O, seq_len, num_heads);
    } 
    else {
        printf("Unsupported head_dim: %d (only 64 is supported)\n", head_dim);
        return;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Kernel failed: %s\n", cudaGetErrorString(err));
    }

}
