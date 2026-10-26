#include <cuda_runtime.h>
#include <math.h>
#include <float.h>
#include <stdio.h>

#define BR 128
#define BC 128
#define D 64


// 教学版 FlashAttention Forward (1 Thread = 1 Row)
__global__ void flash_attn_fp32_pedagogical_kernel(
    const float* Q, const float* K, const float* V, float* O,
    int seq_len
) {
    // 1. 坐标映射
    int tx = threadIdx.x;              // 当前线程负责 Q 的哪一行 (0 ~ 127)
    int bx = blockIdx.x;               // 当前处理的是哪一个 Q Block
    int num_kv_blocks = seq_len / BC;  // K, V 需要遍历的 Block 数量
    
    // 计算缩放因子 1 / sqrt(d)
    float scale = 1.0f / sqrtf((float)D);

    // 2. 局部寄存器分配 (速度最快，每个线程独享)
    float q_i[D];                      // 存储当前线程负责的 Q 的一行
    float o_i[D] = {0.0f};             // 累加器 O_i，初始化为 0
    float m_i = -FLT_MAX;              // Online Softmax 的最大值 m
    float l_i = 0.0f;                  // Online Softmax 的指数和 l

    // 注意力分数缓存 (128 个 float，存在寄存器中)
    float s_ij[BC]; 

    // 3. 共享内存分配 (SRAM，Block 内所有线程共享)
    // 使用动态共享内存绕过 ptxas 64KB > 48KB 默认上限的编译检查
    extern __shared__ float s_mem[];
    float* s_K = s_mem;
    float* s_V = s_mem + BC * D;

    // 4. 将 Q 的当前行从 HBM 加载到寄存器
    int q_offset = bx * (BR * D) + tx * D;
    for (int d = 0; d < D; ++d) {
        q_i[d] = Q[q_offset + d];
    }

    // 5. 外层循环：固定 Q_i，遍历所有的 K_j 和 V_j
    for (int kv_idx = 0; kv_idx < num_kv_blocks; ++kv_idx) {
        
        // --- 步骤 A: 协作将 K_j 和 V_j 加载到 SRAM ---
        int kv_offset = kv_idx * (BC * D) + tx * D;
        for (int d = 0; d < D; ++d) {
            s_K[tx * D + d] = K[kv_offset + d];
            s_V[tx * D + d] = V[kv_offset + d];
        }
        __syncthreads();

        // --- 步骤 B: 计算 S = Q_i * K_j^T ---
        for (int j = 0; j < BC; ++j) {
            float sum = 0.0f;
            for (int d = 0; d < D; ++d) {
                sum += q_i[d] * s_K[j * D + d];
            }
            s_ij[j] = sum * scale;
        }

        // --- 步骤 C: Online Softmax ---
        float m_new = m_i;
        for (int j = 0; j < BC; ++j) {
            m_new = fmaxf(m_new, s_ij[j]);
        }

        float d_l = expf(m_i - m_new);
        float l_new = l_i * d_l;

        for (int j = 0; j < BC; ++j) {
            float p = expf(s_ij[j] - m_new);
            s_ij[j] = p;     
            l_new += p;      
        }

        // --- 步骤 D: 计算 O_i = O_i * d_l + P_ij * V_j ---
        for (int d = 0; d < D; ++d) {
            o_i[d] *= d_l;
            for (int j = 0; j < BC; ++j) {
                o_i[d] += s_ij[j] * s_V[j * D + d]; 
            }
        }

        m_i = m_new;
        l_i = l_new;
        __syncthreads();
    }

    // 6. 最终的归一化与写回 HBM
    int o_offset = bx * (BR * D) + tx * D;
    for (int d = 0; d < D; ++d) {
        O[o_offset + d] = o_i[d] / l_i;
    }
}

// --- Host 端启动函数 ---
// 这个函数会被 flash_attn_api.cpp 调用
void launch_flash_attn_fp32_pedagogical(
    const float* Q, const float* K, const float* V, float* O, int seq_len
) {
    int num_blocks = seq_len / BR; 

    // 1. 计算所需的动态共享内存大小 (字节)
    // s_K 和 s_V 分别是 BC * D 个 float，每个 float 占 4 字节
    int shared_mem_bytes = 2 * BC * D * sizeof(float);

    // 2. 解除 48KB 的共享内存物理限制，申请高达 shared_mem_bytes 的空间
    cudaFuncSetAttribute(
        flash_attn_fp32_pedagogical_kernel, 
        cudaFuncAttributeMaxDynamicSharedMemorySize, 
        shared_mem_bytes
    );

    // 3. 启动 Kernel (注意第三个参数 shared_mem_bytes)
    flash_attn_fp32_pedagogical_kernel<<<num_blocks, 128, shared_mem_bytes>>>(
        Q, K, V, O, seq_len
    );
    
    // 4. 捕获启动错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA Kernel failed: %s\n", cudaGetErrorString(err));
    }
}
