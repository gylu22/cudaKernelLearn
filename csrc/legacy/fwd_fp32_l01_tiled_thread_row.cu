#include <cuda_runtime.h>
#include <stdio.h>
#include <float.h>


#define FLOAT4(v) (reinterpret_cast<float4*>(&(v))[0])



template<const int BR=64, const int BC=64, int D=64>
__global__ void flash_attention_v2_fp32_forward_kernel(
    float* Q,  float* K,  float* V, float* O, int seq_len, int num_heads){
    
    int head_offset = seq_len * D; // 每个头的起始位置
    int batch_offset = blockIdx.z * num_heads * head_offset;
    int q_offset = batch_offset + blockIdx.y * head_offset + blockIdx.x * BR * D; // Q 的起始位置

    float* q_ptr = Q + q_offset; // 当前 block 处理的 Q 的起始位置
    float* k_ptr = K + batch_offset + blockIdx.y * head_offset;
    float* v_ptr = V + batch_offset + blockIdx.y * head_offset;
    float* o_ptr = O + q_offset; // 输出的起始位置

    int tid = threadIdx.x; // 线程 ID

    extern __shared__ float shared_mem[]; // 共享内存用于存储 K 和 V
    float* q_shared = shared_mem; // Q 的共享内存
    float* k_shared = q_shared + BR * D; // K 的共享内存
    float* v_shared = k_shared + BC * D; // V 的共享内存
    float s_Q[BC] = {0.f};
    float s_O[D] = {0.f};
    float s_m = -FLT_MAX;
    float s_l = 0.0f;
    float scale = 1.0f / sqrtf((float)D);
    // Q: global -> shared
    int step = blockDim.x / (D / 4); // 4 循环16次
    int row = tid / (D / 4); // 当前线程处理的 Q 的行
    int col = tid % (D / 4) * 4; // 当前线程处理的起始偏移量
    for (int q_idx =0; q_idx < BR; q_idx += step) {
        int tmp = (q_idx + row) * D + col;
        if (q_idx + row < BR) {
            FLOAT4(q_shared[tmp]) = FLOAT4(q_ptr[tmp]);
        }
    }
    for (int kv_idx=0;kv_idx<seq_len;kv_idx+=BC){
        // K V: global -> shared
        for (int kv_idx_inner=0;kv_idx_inner<BC;kv_idx_inner+=step){
            int tmp1 = (kv_idx_inner + row) * D + col;
            int tmp2 = (kv_idx + kv_idx_inner + row) * D + col;
            if (kv_idx + kv_idx_inner + row < seq_len) {
                FLOAT4(k_shared[tmp1]) = FLOAT4(k_ptr[tmp2]);
                FLOAT4(v_shared[tmp1]) = FLOAT4(v_ptr[tmp2]);
            }
        }
        __syncthreads();
        // step 1
        for (int k_idx=0;k_idx<BC;k_idx++){
            float dot = 0.f;
            for (int d=0;d<D;d++){
                dot += q_shared[tid * D + d] * k_shared[k_idx * D + d];
            }
            s_Q[k_idx] = dot * scale;
        }
         // step 2
        // 计算 online softmax, 更新 s_m 和 s_l
        float prev_s_m = s_m; // 记录之前的 s_m
        for (int i=0;i<BC;i++){
            s_m = fmaxf(s_Q[i], s_m);   
        }
        // step 3
        float decay = expf(prev_s_m - s_m); // 计算分母的缩放因子
        s_l = s_l * decay; // 更新 s_l
        for (int i=0;i<D;i++){
            s_O[i] *= decay; // 计算分子，并乘以缩放因子
        }
        // step 4
        // 计算 attention输出
        for (int i=0;i<BC;i++){
            float p = expf(s_Q[i] - s_m); 
            s_l += p; // 更新 s_l
            for (int d=0;d<D;d++){
                s_O[d] += p * v_shared[i * D + d]; 
            }
        }
        __syncthreads();
    }
    for (int i=0;i<D;i++){
        o_ptr[tid * D + i] = s_O[i] / s_l; // 输出结果
    }

}
   

void flash_attention_v2_fp32_forward_host(
    float* Q,  float* K,  float* V, float* O, int bs, int num_heads, int seq_len, int head_dim){

    const int BR = 64; // 每个 block 处理的序列长度
    const int BC = 64; // 每个 block 处理的 K/V 列数
    int num_blocks = seq_len / BR; // 每个头的 block 数量

    dim3 grid(num_blocks, num_heads, bs);
    dim3 block(BR);

    // 根据 head_dim 分发到正确的模板实例
    if (head_dim == 64) {
        const int D = 64;
        int shared_mem_bytes = (BC * D * 2 + BR * D) * sizeof(float);
        cudaFuncSetAttribute(
            flash_attention_v2_fp32_forward_kernel<BR, BC, D>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shared_mem_bytes);
        flash_attention_v2_fp32_forward_kernel<BR, BC, D><<<grid, block, shared_mem_bytes>>>(Q, K, V, O, seq_len, num_heads);
    } else if (head_dim == 128) {
        const int D = 128;
        int shared_mem_bytes = (BC * D * 2 + BR * D) * sizeof(float);
        cudaFuncSetAttribute(
            flash_attention_v2_fp32_forward_kernel<BR, BC, D>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            shared_mem_bytes);
        flash_attention_v2_fp32_forward_kernel<BR, BC, D><<<grid, block, shared_mem_bytes>>>(Q, K, V, O, seq_len, num_heads);
    } else {
        printf("Unsupported head_dim: %d (only 64 and 128 are supported)\n", head_dim);
        return;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Kernel failed: %s\n", cudaGetErrorString(err));
    }

}
