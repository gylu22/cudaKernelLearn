# CUDA Kernel 学习路线

本仓库用两个算子串起同一组能力：SGEMM 负责建立矩阵分块、数据复用和流水线
直觉；FlashAttention 把这些能力扩展到 online softmax、跨 tile 状态与融合算子。

如果已经一段时间没有接触 CUDA，建议先完成下面的基础复习：

```text
GPU/SM/warp 执行模型
  → register/shared/L1/L2/global 内存层次
  → barrier/fence/atomic/stream
  → latency/throughput/TLP/ILP/MLP/Roofline
  → 数值精度与 correctness
  → kernel 开发与 profiling 闭环
```

对应文档：

1. [GPU 硬件与执行模型](fundamentals/gpu-architecture.md)
2. [GPU 内存层次](fundamentals/memory-hierarchy.md)
3. [同步、Stream 与并发](fundamentals/execution-and-synchronization.md)
4. [Kernel 性能模型](fundamentals/performance-model.md)
5. [数值精度](fundamentals/numerical-precision.md)
6. [Kernel 开发方法](fundamentals/kernel-development.md)

## 阶段 0：先建立可复现基线

完成标准不是“kernel 能跑”，而是：

- 明确输入契约、dtype、layout、支持的 shape 和最低 compute capability；
- 有 PyTorch reference、边界尺寸、错误输入和 current-stream 测试；
- benchmark 将 warmup、同步和计时范围写清楚；
- 编译记录包含目标架构、寄存器、spill 和 shared memory；
- 每个优化保留为独立 backend，不覆盖基线。

## 阶段 1：SGEMM 数据通路

按当前 backend 顺序学习：

1. `simt_fp32_00_naive`：线程/Block/Grid 映射和合并访问。
2. `simt_fp32_01_smem_tiled`：global → shared → register 的分块复用。
3. `simt_fp32_02_thread_tile_vec4`：thread tile、ILP、寄存器压力和向量化访问。
4. `simt_fp32_03_transposed_smem_vec4`：shared-memory layout 与 bank conflict。
5. `simt_fp32_04_double_buffer`：先验证逻辑双缓冲，再引入异步拷贝。
6. `simt_fp32_05_padded_smem`：用 NCU 验证 padding 是否真的减少事务，而不是从
   名称推断效果。

每一步至少回答：计算量是否变化、global/shared 事务是否变化、寄存器和 occupancy
如何变化、kernel duration 是否稳定下降。

## 阶段 2：异步流水线与 Tensor Core

学习顺序建议为：

```text
同步 global→shared copy
  → 两级逻辑流水线
  → cp.async / cuda::memcpy_async
  → 多级 pipeline
  → WMMA micro-kernel
  → 明确 layout 的 mma.sync / CUTLASS-CuTe
```

异步 API 只解决数据搬运和依赖表达，不自动保证 overlap。需要同时检查 stage 数量、
每 stage 计算时长、shared-memory 占用、寄存器压力和 scheduler issue efficiency。

## 阶段 3：FlashAttention

1. 用普通 attention 明确 `S=QK^T/sqrt(d)`、softmax 和 `PV` 的 reference。
2. 推导 online softmax 的 running max、normalizer 和输出重标定。
3. 保持数学与 launch geometry 可比，研究 score 放寄存器还是 shared memory。
4. 减小每线程 fragment，使用 warp reduction，观察 occupancy 与 ILP 的权衡。
5. 分别验证 Tensor Core QK、softmax、Tensor Core PV，再组合。
6. 之后再扩展 causal、FP16/BF16、不同 Q/K 长度和 backward。

## 推荐阅读顺序

1. [CUDA 基础概念地图](fundamentals/cuda.md)
2. 上面的六份基础专题
3. [SGEMM 路线](sgemm/index.md)
4. [SGEMM 优化思路与性能差距分析](sgemm/optimization-guide.md)
5. [Profiling 工作流](profiling.md)与[案例推演](profiling/09-case-studies.md)
6. [异步拷贝](tooling/async-copy.md)与[多级流水线](tooling/multistage-pipeline.md)
7. [FlashAttention 架构](architecture.md)
8. [FlashAttention 优化思路](flash-attention/optimization-guide.md)
9. [CUDA 面试复习](fundamentals/interview-review.md)与
   [Profiling 面试问答](profiling/10-interview-playbook.md)

需要安排实际开发任务时，再分别阅读
[SGEMM 生产化路线](sgemm/production-roadmap.md)与
[FlashAttention 后续路线](flash-attention/roadmap.md)。
