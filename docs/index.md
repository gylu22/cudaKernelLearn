# 文档中心

这套文档围绕“实现一个 kernel，并用证据解释它为什么更快或更慢”组织。建议先读
[学习路线](learning-path.md)，实际调优时从 [Profiling 工作流](profiling.md)
开始。

## 算子路线

- [SGEMM](sgemm/index.md)：活动 backend、尺寸约束与运行方法。
- [SGEMM 优化思路与性能差距分析](sgemm/optimization-guide.md)：现有优化链路、
  shape 敏感性，以及部分矩阵接近 cuBLAS、部分矩阵差距较大的原因。
- [SGEMM 生产化差距与后续路线](sgemm/production-roadmap.md)：从 benchmark
  公平性、一般 shape、kernel family 到 Tensor Core 和发布门槛。
- [SGEMM 历史实现笔记](sgemm/legacy-kernel-notes.md)：保留旧实现的设计意图，并
  修正原 bank-conflict 分析。
- [FlashAttention 架构](architecture.md)：输入契约和 backend 边界。
- [FlashAttention 优化思路](flash-attention/optimization-guide.md)：online
  softmax、现有线程映射、资源瓶颈与逐步优化方法。
- [FlashAttention 后续优化路线](flash-attention/roadmap.md)：从 fragment
  重构、流水线、Tensor Core 到功能覆盖和生产化的分阶段计划。

## CUDA 基础

- [CUDA 基础概念地图](fundamentals/cuda.md)：用于快速回顾执行模型、内存、同步、
  精度和常见优化模式。
- [GPU 硬件与执行模型](fundamentals/gpu-architecture.md)：SM、sub-partition、
  warp scheduler、执行管线、资源驻留和 wave。
- [GPU 内存层次](fundamentals/memory-hierarchy.md)：register/local/shared、
  L1/L2/global、sector、coalescing、bank 与 host-device 数据移动。
- [同步、内存模型、Stream 与并发](fundamentals/execution-and-synchronization.md)：
  barrier、fence、atomic、thread scope、异步错误和 CUDA Graph。
- [Kernel 性能模型](fundamentals/performance-model.md)：latency/throughput、
  TLP/ILP/MLP、Roofline、wave quantization 和优化决策树。
- [数值精度](fundamentals/numerical-precision.md)：浮点格式、舍入、FMA、累加、
  softmax、fast-math、确定性和测试策略。
- [Kernel 开发方法](fundamentals/kernel-development.md)：从 reference、工作映射、
  tiling、边界到 dispatcher 和生产门禁。
- [CUDA 面试复习提纲](fundamentals/interview-review.md)：按概念、原因、权衡和验证
  组织的高频问题。
- [异步拷贝](tooling/async-copy.md)：Ampere `cp.async`/LDGSTS 与
  `cuda::pipeline`。
- [多级流水线](tooling/multistage-pipeline.md)：stage 数量、同步与资源权衡。

## Profiling

0. [Profiling 总工作流](profiling.md)
1. [Profiling 方法论](profiling/00-methodology.md)
2. [统一实验基础设施](experiments.md)
3. [可复现实验与热点筛选](profiling/01-workflow.md)
4. [Nsight Systems](profiling/06-nsight-systems.md)
5. [Speed of Light 吞吐分析](profiling/02-speed-of-light.md)
6. [Memory Workload Analysis](profiling/03-memory-workload.md)
7. [Compute、Scheduler 与 Stall](profiling/04-compute-and-scheduler.md)
8. [Occupancy](profiling/occupancy.md)
9. [缓存与未命中](profiling/cache.md)
10. [Roofline 模型](profiling/05-roofline.md)
11. [NCU 指标关系与诊断字典](profiling/08-metric-reference.md)
12. [Source、PTX 与 SASS](profiling/07-source-and-sass.md)
13. [SGEMM/FlashAttention 案例推演](profiling/09-case-studies.md)
14. [Kernel Profiling 面试与实战问答](profiling/10-interview-playbook.md)

## 工程与部署

- [CMake 与 CUDA](tooling/cmake-cuda.md)
- [PyTorch CUDA Extension](tooling/pytorch-extensions.md)
- [ONNX 与 ONNX Runtime](frameworks/onnx-runtime.md)
- [TensorRT](frameworks/tensorrt.md)
- [Elementwise Add 教学示例](examples/elementwise-add.md)
- [CUDA 实验记录模板](experiment-template.md)

## 阅读约定

- 文档中的性能阈值只能作为线索，不能替代同硬件、同输入、同编译参数的测量。
- Nsight Compute 指标和 section 名称会随版本及 GPU 架构变化；优先理解指标含义，
  再在当前报告中搜索对应项。
- “支持”与“更快”是两件事。数据类型、Tensor Core 或异步 API 可用，不代表给定
  shape 上必然获益。
- 每篇技术文档末尾列出主要官方资料；涉及版本差异时以本机工具文档为准。

## 原始笔记

被以上校订版替代的原文无损保存在
[`archive/sgemm-profiling-original/`](archive/sgemm-profiling-original/)。
其中包含过时接口、未经实测的阈值和已知错误，仅用于回看学习过程，不应作为当前
实现说明。
