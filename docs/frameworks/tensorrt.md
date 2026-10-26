# TensorRT 推理与扩展

TensorRT 会针对给定网络、目标 GPU、精度和 shape profile 搜索实现并序列化为 engine。engine 是构建结果，不是通用模型格式。本文以当前 TensorRT 文档为准；旧项目中的 implicit batch、隐式 INT8 校准和 `IPluginV2*` 接口应视为历史方案。

## 1. 基本工作流

典型流程是：

1. 用 ONNX parser 或 Network Definition API 创建网络；
2. 配置 builder、精度、内存池和 optimization profile；
3. 构建并序列化 engine；
4. runtime 反序列化 engine；
5. 为并发执行创建 execution context；
6. 设置动态输入 shape 和张量地址；
7. 在调用方 CUDA stream 上执行 `enqueueV3`。

构建阶段可能做常量折叠、层融合、精度选择和 tactic 搜索。候选 tactic 与 TensorRT 版本、GPU、shape、内存限制和计时噪声有关，所以构建产物和性能记录必须携带环境信息。

## 2. 动态形状与 profile

动态输入必须为每个 optimization profile 指定 `min`、`opt`、`max`。`min` 与 `max` 限定范围，`opt` 是 tactic 优化的代表形状而不是唯一可运行形状。范围过宽可能减少可选 tactic，或让非典型形状性能变差。

profile 应来自真实流量，而不是机械地覆盖所有尺寸。每个并发推理通常使用独立 execution context；切换 profile 和复用 context 时须遵守当前版本 API 约束。

## 3. 精度与量化

当前 TensorRT 使用 strongly typed network 明确张量类型和精度关系。FP16、BF16、TF32、FP8、INT8 或 INT4 是否可用及是否更快，取决于版本、GPU、算子和 shape。

当前推荐用显式 Q/DQ 节点表达量化。TensorRT 11 已移除隐式量化相关接口，早期版本中的 calibrator 工作流不应作为新项目默认方案。部署前要确认 Q/DQ 位置、scale/zero point、精度回退、数值误差和真实性能。

## 4. tactic、timing cache 与内存

builder 在可用实现中选择 tactic。timing cache 能减少重复构建成本并提高相似构建的稳定性，但只能在兼容环境中复用。应记录 GPU、TensorRT/CUDA/驱动版本、builder flags、profile、精度、内存池上限、plugin 版本和 cache 生成条件。

workspace 等内存池配置通常是构建期可用内存上限，不代表运行时必然预留同等大小。运行时还需要权重、activation、execution context 和输入输出缓冲区。

## 5. Plugin

当前自定义层优先使用 `IPluginV3` 系列接口。plugin 通常要实现输出 dtype/format/shape 推导、配置与序列化、workspace 大小、执行函数和 creator 注册。

执行函数必须使用 TensorRT 传入的 CUDA stream，不应擅自 `cudaDeviceSynchronize()`。兼容旧工程时会看到 `IPluginV2DynamicExt`，但它已属于弃用路线；升级时应阅读目标版本迁移指南，不能只替换类名。

## 6. Engine 兼容性

默认情况下，不应假定序列化 engine 能跨 TensorRT 版本或任意 GPU 复用。version-compatible 和 hardware compatibility 选项也有构建约束与性能权衡。

稳妥的发布策略是保留可重建的 ONNX/网络定义，把构建配置、plugin 和版本写入制品元数据，按目标环境构建与校验 engine，并为反序列化失败提供重建或回退路径。

## 7. 性能分析

建议分三层：

1. `trtexec`：检查可构建性、精度、吞吐和延迟；
2. Nsight Systems：观察拷贝、enqueue、kernel 间空洞和并发；
3. Nsight Compute：分析已经定位的关键 kernel。

不要把 TensorRT 的 GPU compute time 与另一个框架的端到端时间直接比较。报告应说明 batch、shape、warmup、并发数、数据位置、同步点和统计分位数。Polygraphy 适合做数值比较、模型缩减和 tactic 调试；TensorRT Engine Explorer（TREX）可查看 engine 图层与性能数据。

TensorRT-LLM 是建立在 TensorRT 之上的 LLM 推理栈，额外处理量化、KV cache、并行和服务运行时。普通 TensorRT plugin 与 engine 生命周期仍有价值，但两者的构建和部署接口不能混用。

## 8. 官方资料

- [TensorRT documentation](https://docs.nvidia.com/deeplearning/tensorrt/latest/)
- [Dynamic shapes](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-dynamic-shapes.html)
- [Quantized types](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/work-quantized-types.html)
- [Custom layers](https://docs.nvidia.com/deeplearning/tensorrt/latest/inference-library/extending-custom-layers.html)
- [Command-line programs](https://docs.nvidia.com/deeplearning/tensorrt/latest/reference/command-line-programs.html)
- [Best practices](https://docs.nvidia.com/deeplearning/tensorrt/latest/performance/best-practices.html)

