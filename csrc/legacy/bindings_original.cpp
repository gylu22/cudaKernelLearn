#include <torch/extension.h>
#include <cuda_runtime.h>
#include <cstdio>

// ============================================================================
// GPU Profiling 信息打印函数
// 打印所有对 CUDA kernel 调优/occupancy 计算有帮助的硬件参数
// ============================================================================
void print_gpu_profiling_info() {
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error (getDeviceCount): %s\n", cudaGetErrorString(err));
        return;
    }

    int currentDevice;
    cudaGetDevice(&currentDevice);

    printf("\n");
    printf("╔══════════════════════════════════════════════════════════════════╗\n");
    printf("║                GPU Profiling Information                         ║\n");
    printf("╚══════════════════════════════════════════════════════════════════╝\n");
    printf("  Device count: %d  (current: %d)\n\n", deviceCount, currentDevice);

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        // ── Identification ──────────────────────────────────────────
        printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        printf("  Device %d: %s\n", i, prop.name);
        printf("  Compute Capability:  %d.%d\n", prop.major, prop.minor);
        printf("  PCI Bus:             %04x:%02x:%02x.0\n",
               prop.pciDomainID, prop.pciBusID, prop.pciDeviceID);

        // ── Compute Resources (occupancy 计算核心) ──────────────────
        printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        printf("  [Compute Resources]\n");
        printf("  SMs (Multiprocessors):           %d\n", prop.multiProcessorCount);
        printf("  Max Threads per SM:              %d\n", prop.maxThreadsPerMultiProcessor);
        printf("  Max Threads per Block:           %d\n", prop.maxThreadsPerBlock);
        printf("  Max Blocks per SM:               %d\n", prop.maxBlocksPerMultiProcessor);
        printf("  Warp Size:                       %d\n", prop.warpSize);
        printf("  Max Warps per SM:                %d\n",
               prop.maxThreadsPerMultiProcessor / prop.warpSize);
        printf("  Max Block Dim:                   (%d, %d, %d)\n",
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("  Max Grid Dim:                    (%d, %d, %d)\n",
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);

        // ── Memory Hierarchy ────────────────────────────────────────
        printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        printf("  [Memory Hierarchy]\n");
        printf("  Global Memory:                   %.2f GB\n",
               prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));

        // Shared Memory —— kernel 调优最关键参数之一
        printf("  ══ Shared Memory ══\n");
        printf("  SMEM per Block (default):        %5zu KB\n",
               prop.sharedMemPerBlock / 1024);
        printf("  SMEM per Block (opt-in):         %5zu KB  ← cudaFuncSetAttribute\n",
               prop.sharedMemPerBlockOptin / 1024);
        printf("  SMEM per SM:                     %5zu KB  ← 限制 blocks/SM 的上限\n",
               prop.sharedMemPerMultiprocessor / 1024);

        // L1 / L2 Cache
        printf("  ══ Cache ══\n");
        printf("  L2 Cache Size:                   %5d KB  (%.2f MB)\n",
               prop.l2CacheSize / 1024, prop.l2CacheSize / (1024.0 * 1024.0));
        // L1: 与 SMEM 共享片上 SRAM，具体 split 取决于 cache 配置
        // 多数架构 (SM 8.0+) SMEM/L1 统一 128 KB/SM
        printf("  L1 Cache:                        片上与 SMEM 共用 (见 SMEM per SM)\n");

        // Register file
        printf("  ══ Register File ══\n");
        printf("  Registers per Block:             %d\n", prop.regsPerBlock);
        printf("  Registers per SM:                %d  ← 限制 blocks/SM 的上限\n",
               prop.regsPerMultiprocessor);
        printf("  Registers per Thread (max):      %d\n",
               prop.regsPerBlock / prop.maxThreadsPerBlock);
        printf("  注: 寄存器分配粒度 = 256 regs/block (CC 5.0+)\n");

        printf("  Constant Memory:                 %zu KB\n", prop.totalConstMem / 1024);

        // ── Memory Bandwidth ─────────────────────────────────────────
        printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        printf("  [Memory Bandwidth]\n");
        printf("  Memory Bus Width:                %d bits  (= %d bytes/transaction)\n",
               prop.memoryBusWidth, prop.memoryBusWidth / 8);
        printf("  Memory Clock Rate:               %.2f GHz\n",
               prop.memoryClockRate / 1e6);
        // memoryClockRate 单位 kHz: kHz × (bus_width/8) × 2(DDR) × 1000 / 1e9 = GB/s
        double bw = 2.0 * (prop.memoryClockRate * 1000.0) * (prop.memoryBusWidth / 8.0) / 1e9;
        printf("  Peak Memory Bandwidth:           %.1f GB/s  ← DRAM 理论峰值\n", bw);
        printf("  GPU Clock Rate:                  %.2f GHz\n", prop.clockRate / 1e6);

        // ── Occupancy Calculator ─────────────────────────────────────
        printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        printf("  [Occupancy Calculator — 128-thread blocks]\n");

        // SMEM → blocks/SM
        {
            int smem_kb_vals[] = {8, 16, 24, 32, 40, 48, 52, 56, 64, 72, 80, 88, 96, 112, 128, -1};
            printf("  ┌────────────────────────────────────────────────────┐\n");
            printf("  │ SMEM/block → blocks/SM → threads/SM → occupancy   │\n");
            printf("  ├────────────────────────────────────────────────────┤\n");
            for (int j = 0; smem_kb_vals[j] >= 0; j++) {
                int skb = smem_kb_vals[j];
                size_t sbytes = (size_t)skb * 1024;
                if (sbytes > prop.sharedMemPerMultiprocessor) break;
                int blk = (int)(prop.sharedMemPerMultiprocessor / sbytes);
                if (blk > prop.maxBlocksPerMultiProcessor) blk = prop.maxBlocksPerMultiProcessor;
                int thr = blk * 128;
                if (thr > prop.maxThreadsPerMultiProcessor) thr = prop.maxThreadsPerMultiProcessor;
                printf("  │ %4d KB          → %4d          → %6d      → %3d%%  │\n",
                       skb, blk, thr, thr * 100 / prop.maxThreadsPerMultiProcessor);
            }
            printf("  └────────────────────────────────────────────────────┘\n");
        }

        // Registers → blocks/SM
        {
            int reg_vals[] = {24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 112, 128, 144, 160, 192, 224, -1};
            int alloc_unit = 256;  // register allocation granularity per block
            printf("  ┌──────────────────────────────────────────────────────────┐\n");
            printf("  │ regs/thread → block alloc → blocks/SM → occupancy       │\n");
            printf("  ├──────────────────────────────────────────────────────────┤\n");
            for (int j = 0; reg_vals[j] >= 0; j++) {
                int rpt = reg_vals[j];
                // round up to allocation unit
                int reg_per_blk = ((rpt * 128 + alloc_unit - 1) / alloc_unit) * alloc_unit;
                int blk = prop.regsPerMultiprocessor / reg_per_blk;
                if (blk > prop.maxBlocksPerMultiProcessor) blk = prop.maxBlocksPerMultiProcessor;
                if (blk == 0) {
                    printf("  │ %4d           → %6d     →  ❌ 无法启动              │\n",
                           rpt, reg_per_blk);
                } else {
                    int thr = blk * 128;
                    if (thr > prop.maxThreadsPerMultiProcessor) thr = prop.maxThreadsPerMultiProcessor;
                    printf("  │ %4d           → %6d     → %6d      → %3d%%        │\n",
                           rpt, reg_per_blk, blk, thr * 100 / prop.maxThreadsPerMultiProcessor);
                }
            }
            printf("  └──────────────────────────────────────────────────────────┘\n");
        }

        // 综合限制说明
        printf("  Occupancy = min(threads/SM 限制, SMEM/SM 限制, regs/SM 限制)\n");
        printf("  你的 kernel 使用 128 线程 + %.1f KB SMEM + N regs/thread\n",
               (3.0f * 64 * 68 * sizeof(float)) / 1024.0f);  // rough estimate

        // ── Capabilities ─────────────────────────────────────────
        printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n");
        printf("  [Capabilities]\n");
        printf("  Concurrent Kernels:              %s\n",
               prop.concurrentKernels ? "Yes" : "No");
        printf("  Copy-Engine Overlap:             %s\n",
               prop.deviceOverlap ? "Yes" : "No");
        printf("  Async Engine Count:              %d\n", prop.asyncEngineCount);
        printf("  Unified Memory:                  %s\n",
               prop.managedMemory ? "Yes" : "No");
        printf("  Cooperative Launch:              %s\n",
               prop.cooperativeLaunch ? "Yes" : "No");
        printf("  ECC Enabled:                     %s\n",
               prop.ECCEnabled ? "Yes" : "No");
        printf("  Compute Mode:                    ");
        switch (prop.computeMode) {
            case cudaComputeModeDefault:          printf("Default\n"); break;
            case cudaComputeModeExclusive:        printf("Exclusive (single context)\n"); break;
            case cudaComputeModeProhibited:       printf("Prohibited\n"); break;
            case cudaComputeModeExclusiveProcess: printf("Exclusive Process\n"); break;
            default: printf("Unknown\n");
        }
        printf("\n");
    }
    printf("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n");
}

// 声明定义在 .cu 文件中的启动函数
void launch_flash_attn_fp32_pedagogical(
    const float* Q, const float* K, const float* V, float* O, int seq_len
);


void flash_attention_v2_fp32_forward_host(
    float* Q,  float* K, float* V, float* O, 
    int bs, int num_heads, int seq_len, int head_dim
);

void flash_attention_v2_fp32_warp_forward_host(
    float* Q,  float* K, float* V, float* O, 
    int bs, int num_heads, int seq_len, int head_dim
);

void flash_attention_v2_fp32_optimized_host(
    const float* Q, const float* K, const float* V, float* O,
    int batch_size, int num_heads, int seq_len, int head_dim);

void flash_attention_v2_fp32_gemm_dynamic_host(
    float* Q,  float* K,  float* V, float* O,
    int batch_size, int num_heads, int seq_len, int head_dim);

void flash_attention_v2_fp32_gemm_kv_reuse_host(
    float* Q,  float* K,  float* V, float* O,
    int batch_size, int num_heads, int seq_len, int head_dim);

// PyTorch 的前向传播接口
torch::Tensor flash_attn_naive(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    // 1. 安全性检查：确保输入是在 GPU 上的连续 float32 张量
    TORCH_CHECK(Q.is_cuda() && Q.is_contiguous(), "Q must be a CUDA contiguous tensor");
    TORCH_CHECK(K.is_cuda() && K.is_contiguous(), "K must be a CUDA contiguous tensor");
    TORCH_CHECK(V.is_cuda() && V.is_contiguous(), "V must be a CUDA contiguous tensor");
    TORCH_CHECK(Q.scalar_type() == torch::kFloat32, "Only float32 is supported");

    int seq_len = Q.size(0);
    
    // 2. 分配输出张量 O 的内存
    auto O = torch::zeros_like(Q);

    // 3. 获取底层数据指针并启动 CUDA Kernel
    launch_flash_attn_fp32_pedagogical(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        O.data_ptr<float>(),
        seq_len
    );

    return O;
}


torch::Tensor flash_attn_v2(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    // 1. 安全性检查：确保输入是在 GPU 上的连续 float32 张量
    TORCH_CHECK(Q.is_cuda() && Q.is_contiguous(), "Q must be a CUDA contiguous tensor");
    TORCH_CHECK(K.is_cuda() && K.is_contiguous(), "K must be a CUDA contiguous tensor");
    TORCH_CHECK(V.is_cuda() && V.is_contiguous(), "V must be a CUDA contiguous tensor");
    TORCH_CHECK(Q.scalar_type() == torch::kFloat32, "Only float32 is supported");

   
    int bs = Q.size(0);
    int num_heads = Q.size(1);
    int seq_len = Q.size(2);
    int head_dim = Q.size(3);

    // 2. 分配输出张量 O 的内存
    auto O = torch::zeros_like(Q);

    // 3. 获取底层数据指针并启动 CUDA Kernel
    flash_attention_v2_fp32_forward_host(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        O.data_ptr<float>(),
        bs,
        num_heads,
        seq_len,
        head_dim
    );

    return O;
}


torch::Tensor flash_attn_v2_warp(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    // 1. 安全性检查：确保输入是在 GPU 上的连续 float32 张量
    TORCH_CHECK(Q.is_cuda() && Q.is_contiguous(), "Q must be a CUDA contiguous tensor");
    TORCH_CHECK(K.is_cuda() && K.is_contiguous(), "K must be a CUDA contiguous tensor");
    TORCH_CHECK(V.is_cuda() && V.is_contiguous(), "V must be a CUDA contiguous tensor");
    TORCH_CHECK(Q.scalar_type() == torch::kFloat32, "Only float32 is supported");

   
    int bs = Q.size(0);
    int num_heads = Q.size(1);
    int seq_len = Q.size(2);
    int head_dim = Q.size(3);

    // 2. 分配输出张量 O 的内存
    auto O = torch::zeros_like(Q);

    // 3. 获取底层数据指针并启动 CUDA Kernel
    flash_attention_v2_fp32_warp_forward_host(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        O.data_ptr<float>(),
        bs,
        num_heads,
        seq_len,
        head_dim
    );

    return O;
}



torch::Tensor flash_attn_v2_gemm(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    // 1. 安全性检查
    TORCH_CHECK(Q.is_cuda() && Q.is_contiguous(), "Q must be a CUDA contiguous tensor");
    TORCH_CHECK(K.is_cuda() && K.is_contiguous(), "K must be a CUDA contiguous tensor");
    TORCH_CHECK(V.is_cuda() && V.is_contiguous(), "V must be a CUDA contiguous tensor");
    TORCH_CHECK(Q.scalar_type() == torch::kFloat32, "Only float32 is supported");

    int bs = Q.size(0);
    int num_heads = Q.size(1);
    int seq_len = Q.size(2);
    int head_dim = Q.size(3);

    // 2. 分配输出张量 O
    auto O = torch::zeros_like(Q);

    // 3. 启动 Dynamic Kernel (Warp-level + 动态共享内存)
    flash_attention_v2_fp32_gemm_dynamic_host(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        O.data_ptr<float>(),
        bs,
        num_heads,
        seq_len,
        head_dim
    );

    return O;
}


torch::Tensor flash_attn_v2_gemm_kv_reuse(torch::Tensor Q, torch::Tensor K, torch::Tensor V) {
    // 1. 安全性检查
    TORCH_CHECK(Q.is_cuda() && Q.is_contiguous(), "Q must be a CUDA contiguous tensor");
    TORCH_CHECK(K.is_cuda() && K.is_contiguous(), "K must be a CUDA contiguous tensor");
    TORCH_CHECK(V.is_cuda() && V.is_contiguous(), "V must be a CUDA contiguous tensor");
    TORCH_CHECK(Q.scalar_type() == torch::kFloat32, "Only float32 is supported");

    int bs = Q.size(0);
    int num_heads = Q.size(1);
    int seq_len = Q.size(2);
    int head_dim = Q.size(3);

    // 2. 分配输出张量 O
    auto O = torch::zeros_like(Q);

    // 3. 启动 Dynamic Kernel (Warp-level + 动态共享内存)
    flash_attention_v2_fp32_gemm_kv_reuse_host(
        Q.data_ptr<float>(),
        K.data_ptr<float>(),
        V.data_ptr<float>(),
        O.data_ptr<float>(),
        bs,
        num_heads,
        seq_len,
        head_dim
    );

    return O;
}






// 4. 将 C++ 函数绑定为 Python 模块
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    // 模块导入时自动打印 GPU profiling 信息
    print_gpu_profiling_info();

    m.def("flash_attn_naive", &flash_attn_naive, "Pedagogical FlashAttention Naive (FP32)");
    m.def("flash_attn_v2", &flash_attn_v2, "FlashAttention v2 (FP32)");
    m.def("flash_attn_v2_warp", &flash_attn_v2_warp, "FlashAttention v2 Warp (FP32)");
    m.def("flash_attn_v2_gemm", &flash_attn_v2_gemm, "FlashAttention v2 Dynamic (FP32, Warp-level, Dynamic SMEM)");
    m.def("flash_attn_v2_gemm_kv_reuse", &flash_attn_v2_gemm_kv_reuse, "FlashAttention v2 Dynamic KV Reuse (FP32, Warp-level, Dynamic SMEM)");

    // GPU Profiling 工具函数
    m.def("print_gpu_info", &print_gpu_profiling_info, "Print detailed GPU profiling info (SM count, SMEM, L1/L2 cache, bandwidth, occupancy calculator)");
}
