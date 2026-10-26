# ONNX 与 ONNX Runtime

本文区分两个经常被混用的概念：**ONNX** 是模型交换格式和算子规范；**ONNX Runtime（ORT）** 是加载、优化并执行 ONNX 模型的推理运行时。

## 1. 模型结构

一个 ONNX `ModelProto` 主要包含计算图 `GraphProto`、算子节点 `NodeProto`、权重或常量 `TensorProto`，以及输入输出的类型和形状信息。

IR version 描述模型容器能力，opset version 描述相应算子域的算子语义。导出端、目标 Execution Provider 和运行时都必须支持模型使用的算子及版本，不能只看 ONNX 文件是否能被解析。大权重可以通过 external data 存在模型文件之外；迁移或发布时必须同时携带这些文件，并验证相对路径。

## 2. 导出后的最小验证

不要把“导出成功”当作“可部署”。建议依次检查：

1. 用 `onnx.checker.check_model` 做结构检查；
2. 记录输入名、dtype、静态/动态维度和 opset；
3. 用 ORT CPU Execution Provider 跑一个小样例；
4. 与源框架比较多个形状、边界值和随机输入；
5. 再切换 CUDA、TensorRT 等目标 Execution Provider；
6. 保存误差指标以及失败模型和输入。

shape inference 能补全一部分中间张量信息，但对动态控制流、自定义算子或依赖运行时值的形状并不完整。它是辅助检查，不是正确性证明。

## 3. Execution Provider

ORT 按注册顺序尝试把图划分给 Execution Provider（EP）。某个 EP 不支持的节点可以回退到后续 EP：

```python
providers = [
    ("TensorrtExecutionProvider", {
        "trt_engine_cache_enable": True,
        "trt_fp16_enable": True,
    }),
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
]
session = ort.InferenceSession("model.onnx", providers=providers)
```

这不保证整张图都在 TensorRT 上运行。部署时应检查实际 provider、图划分、回退节点和数据搬运；频繁的 provider 边界可能抵消算子加速收益。

TensorRT engine cache 与 TensorRT 版本、GPU、模型、构建选项和动态 shape profile 有关。环境或 profile 改变后，应按 ORT 文档要求重新生成或失效缓存。

## 4. 性能优化顺序

推荐依次检查：

1. 输入输出是否在每次推理中往返 CPU/GPU；
2. 是否出现意外的 CPU fallback；
3. 动态 shape/profile 是否覆盖真实负载；
4. CPU EP 的 intra-op/inter-op 线程是否匹配部署环境；
5. 常量折叠、节点消除和融合是否生效；
6. 最后再分析或实现自定义算子。

GPU 推理可使用 I/O Binding 让输入输出直接绑定设备内存。CPU 调优应结合 NUMA、亲和性和并行执行模式测量，不能简单把线程数设为逻辑核心数。ORT 可输出 JSON profiling 数据，分析时应同时看算子耗时、provider、shape 和数据搬运。

## 5. 量化

线性量化通常写成：

```text
real_value = scale × (quantized_value - zero_point)
```

ORT 支持两类常见表示：

- **QDQ**：用 `QuantizeLinear` / `DequantizeLinear` 显式表示量化边界；
- **QOperator**：使用量化版本的算子。

静态量化使用校准数据提前计算激活范围，ORT 提供 MinMax、Entropy 和 Percentile 等校准方法；动态量化在运行时计算部分参数。量化验证至少应覆盖输出误差、逐层误差、目标 EP 是否真正执行量化算子、端到端性能和校准数据代表性。

“INT8 一定更快”并不成立。数据重排、Q/DQ 节点、fallback 和硬件能力都可能改变结果。

## 6. 自定义算子与 CUDA kernel

ORT custom operator 的边界层应检查 dtype、device、rank 和 shape，获取 ORT 提供的 CUDA stream，并在该 stream 上启动 kernel，不能用设备级同步掩盖依赖问题。

如果目标只是让一段子图由 TensorRT 执行，先确认 TensorRT EP 的支持范围；如果需要 TensorRT plugin，还要保证 plugin 在构建 engine 与运行时都已注册。

## 7. 官方资料

- [ONNX IR specification](https://onnx.ai/onnx/repo-docs/IR.html)
- [ONNX Runtime Execution Providers](https://onnxruntime.ai/docs/execution-providers/)
- [ONNX Runtime performance tuning](https://onnxruntime.ai/docs/performance/)
- [ONNX Runtime threading](https://onnxruntime.ai/docs/performance/tune-performance/threading.html)
- [ONNX Runtime quantization](https://onnxruntime.ai/docs/performance/model-optimizations/quantization.html)
- [TensorRT Execution Provider](https://onnxruntime.ai/docs/execution-providers/TensorRT-ExecutionProvider.html)

