#include <cuda_runtime.h>
#include <cmath>
#include <cfloat>
#include <cstdio>

// ================= 【修复：清晰且安全的 float4 读写宏】 =================
// 显式区分：LOAD 用于右值读取（带 const 优化），STORE 用于左值写入
#define LOAD_FLOAT4(ptr)  (*reinterpret_cast<const float4*>(&(ptr)))
#define STORE_FLOAT4(ptr) (*reinterpret_cast<float4*>(&(ptr)))

template<const int BR=64, const int BC=64, const int D=64>
__global__ void flash_attention_v2_fp32_dynamic_kernel(
    const float* __restrict__ Q,  
    const float* __restrict__ K,  
    const float* __restrict__ V, 
    float* __restrict__ O, 
    int seq_len, 
    int num_heads) 
{
    int batch_id = blockIdx.z;
    int head_id = blockIdx.y;
    int q_block_id = blockIdx.x; 

    int head_stride = seq_len * D;
    int batch_stride = num_heads * head_stride;
    int base_offset = batch_id * batch_stride + head_id * head_stride;
    
    const float* q_block_ptr = Q + base_offset + q_block_id * BR * D;
    const float* k_ptr       = K + base_offset;
    const float* v_ptr       = V + base_offset;
    float* o_block_ptr       = O + base_offset + q_block_id * BR * D;

    // ================= 【动态共享内存分配】 =================
    extern __shared__ float shared_mem_pool[];
    
    float* smem_Q = shared_mem_pool;             // 大小: BR * D
    float* smem_K = shared_mem_pool + BR * D;     // 大小: BC * D
    float* smem_V = smem_K + BC * D;             // 大小: BC * D

    int tid = threadIdx.x;        
    int warp_id = tid / 32;       
    int lane_id = tid % 32;       
    const int NUM_WARPS = 4; // 128 线程 / 32 = 4 Warps

    constexpr int ROWS_PER_WARP = BR / NUM_WARPS; // 64 / 4 = 16 行
    int warp_row_start = warp_id * ROWS_PER_WARP;

    // 线程本地状态寄存器（常驻寄存器，避免显存级延迟）
    float local_m[ROWS_PER_WARP];
    float local_l[ROWS_PER_WARP];
    float local_o_acc[ROWS_PER_WARP][2];

    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        local_m[r] = -FLT_MAX;
        local_l[r] = 0.0f;
        local_o_acc[r][0] = 0.0f;
        local_o_acc[r][1] = 0.0f;
    }

    float scale = 1.0f / sqrtf(static_cast<float>(D));

    // 协同加载 Q 到 Shared Memory
    int q_load_threads = blockDim.x;
    #pragma unroll
    for (int i = tid; i < BR * (D / 4); i += q_load_threads) {
        int r = i / (D / 4);
        int c = (i % (D / 4)) * 4;
        if (q_block_id * BR + r < seq_len) {
            STORE_FLOAT4(smem_Q[r * D + c]) = LOAD_FLOAT4(q_block_ptr[r * D + c]);
        } else {
            STORE_FLOAT4(smem_Q[r * D + c]) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        }
    }
    __syncthreads(); 

    // 外层大循环：遍历所有的 K, V 块 (BC)
    for (int kv_block_id = 0; kv_block_id < seq_len; kv_block_id += BC) {

        // 协同加载 K 和 V 到 Shared Memory
        #pragma unroll
        for (int i = tid; i < BC * (D / 4); i += q_load_threads) {
            int r = i / (D / 4);
            int c = (i % (D / 4)) * 4;
            int global_kv_row = kv_block_id + r;

            if (global_kv_row < seq_len) {
                STORE_FLOAT4(smem_K[r * D + c]) = LOAD_FLOAT4(k_ptr[global_kv_row * D + c]);
                STORE_FLOAT4(smem_V[r * D + c]) = LOAD_FLOAT4(v_ptr[global_kv_row * D + c]);
            } else {
                STORE_FLOAT4(smem_K[r * D + c]) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
                STORE_FLOAT4(smem_V[r * D + c]) = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
            }
        }
        __syncthreads(); 

        // 核心计算阶段
        for (int k_idx = 0; k_idx < BC; ++k_idx) {
            int global_k_row = kv_block_id + k_idx;

            // 预读取当前行 K 到寄存器
            float k_val0 = smem_K[k_idx * D + lane_id];
            float k_val1 = smem_K[k_idx * D + lane_id + 32];

            #pragma unroll
            for (int r = 0; r < ROWS_PER_WARP; ++r) {
                int q_row = warp_row_start + r;

                // 计算局部点积
                float q_val0 = smem_Q[q_row * D + lane_id];
                float q_val1 = smem_Q[q_row * D + lane_id + 32];
                float local_dot = q_val0 * k_val0 + q_val1 * k_val1;

                // Warp 内 32 线程协同规约，计算出当前 q_row 与 k_idx 的完整点积
                #pragma unroll
                for (int offset = 16; offset > 0; offset /= 2) {
                    local_dot += __shfl_down_sync(0xffffffff, local_dot, offset);
                }
                // 广播完整的点积结果给当前 Warp 内的所有线程
                float score = __shfl_sync(0xffffffff, local_dot, 0) * scale;

                // 遮蔽越界算子
                if (global_k_row >= seq_len || (q_block_id * BR + q_row) >= seq_len) {
                    score = -FLT_MAX;
                }

                // Online Softmax 寄存器状态并发更新
                float prev_m = local_m[r];
                local_m[r] = fmaxf(score, local_m[r]);
                
                float decay = expf(prev_m - local_m[r]);
                float p = expf(score - local_m[r]);
                local_l[r] = local_l[r] * decay + p;

                // 累加更新本地的 O 寄存器 (32线程同时为不同的维度出力)
                float v_val0 = smem_V[k_idx * D + lane_id];
                float v_val1 = smem_V[k_idx * D + lane_id + 32];

                local_o_acc[r][0] = local_o_acc[r][0] * decay + p * v_val0;
                local_o_acc[r][1] = local_o_acc[r][1] * decay + p * v_val1;
            }
        }
        __syncthreads(); 
    }

    // 结尾阶段：执行最终的 Softmax 归一化并写回 Global Memory
    #pragma unroll
    for (int r = 0; r < ROWS_PER_WARP; ++r) {
        int q_row = warp_row_start + r;
        int global_q_row = q_block_id * BR + q_row;

        if (global_q_row < seq_len) {
            float final_o0 = local_o_acc[r][0] / local_l[r];
            float final_o1 = local_o_acc[r][1] / local_l[r];

            o_block_ptr[q_row * D + lane_id] = final_o0;
            o_block_ptr[q_row * D + lane_id + 32] = final_o1;
        }
    }
}

// ================= 【Host 端调用接口定义】 =================
void flash_attention_v2_fp32_dynamic_host(
    const float* Q, const float* K, const float* V, float* O, 
    int batch_size, int num_heads, int seq_len, int head_dim) 
{
    if (head_dim != 64) {
        printf("Error: This optimized kernel currently only supports head_dim=64.\n");
        return;
    }

    const int BR = 64;
    const int BC = 64;
    const int D  = 64;

    // 128 线程 (包含 4 个完整的 Warp)
    dim3 block(128); 
    // Grid 映射
    dim3 grid((seq_len + BR - 1) / BR, num_heads, batch_size);

    // 计算动态共享内存大小: (64*64 + 64*64 + 64*64) * 4字节 = 49152 字节 (48 KB)
    int shared_mem_bytes = (BR * D + BC * D + BC * D) * sizeof(float);

    // 设置动态共享内存的高级属性限制（解除硬件默认 48KB 的保护警报）
    cudaFuncSetAttribute(
        flash_attention_v2_fp32_dynamic_kernel<BR, BC, D>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        shared_mem_bytes);

    // 启动 Kernel，第三个参数显式传入共享内存字节数
    flash_attention_v2_fp32_dynamic_kernel<BR, BC, D><<<grid, block, shared_mem_bytes>>>(Q, K, V, O, seq_len, num_heads);
    
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Flash Attention Dynamic Kernel Launch Failed: %s\n", cudaGetErrorString(err));
    }
}
