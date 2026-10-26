# 从 Reference 到生产 Kernel 的开发方法

本文是一套可重复的 kernel 开发流程。目标是每次优化都能回答：改了什么、为什么
可能更快、是否仍然正确、在哪些 shape/硬件上有效。

## 1. 第一步不是写 CUDA

先定义算子契约：

```text
输入/输出 shape:
dtype 与 accumulator:
layout/stride:
支持的 device/architecture:
边界与空维度:
NaN/Inf:
确定性:
允许的误差:
是否需要 backward:
是否允许额外 workspace:
```

没有契约就无法区分优化、功能变化和数值变化。

## 2. 建立可信 reference

reference 应优先可读和高精度：

- 小 shape 用 CPU/FP64；
- 实际 dtype 用 PyTorch/cuBLAS；
- 将复杂融合拆成阶段；
- 保存极端输入与已知答案；
- 不复用待测 kernel 的索引逻辑。

FlashAttention 可以分别验证 QK、stable softmax、PV，再验证 online softmax 融合。

## 3. 写最简单 GPU baseline

baseline 的作用：

- 验证索引和 binding；
- 建立 correctness；
- 观察朴素 global access；
- 成为每次优化的固定对照。

不要一开始混合 vector load、shared tiling、double buffer 和 Tensor Core，否则性能
变化不可归因，bug 也难定位。

## 4. 选择工作映射

逐层回答：

```text
一个 grid 处理什么？
一个 block 处理什么 tile？
一个 warp 处理什么？
一个 lane 处理哪些元素？
数据如何从 global → shared → register？
输出如何写回？
```

映射需要同时满足：

- global coalescing；
- shared bank 友好；
- 足够 block/wave；
- thread tile 提供复用与 ILP；
- tail 可安全处理；
- 同步范围尽可能小。

## 5. Tile 设计

### Block tile

大 tile：

- 数据复用高；
- block 数少；
- shared/register 多；
- tail 浪费可能大。

小 tile：

- grid 并行度高；
- 资源轻；
- 复用较低；
- 地址和边界开销占比可能高。

### Thread tile

大 thread tile 增加 accumulator、ILP 与复用，也提高 register pressure。选择必须用
`registers/thread + occupancy + spill + duration` 共同判断。

### K/stage tile

BK 大可减少循环/barrier 次数，却增加 shared buffer 和 fragment。多 stage 能隐藏
latency，却进一步增加资源。必须 sweep，而不是凭直觉固定。

## 6. 边界策略

常见方案：

1. 全部 predication；
2. 完整 interior fast path + edge kernel；
3. padding/copy 到对齐 buffer；
4. 不支持时 library fallback。

生产实现不应让用户理解内部 tile 才能避免越界。

向量访问必须证明：

- 指针自然对齐；
- 每 lane vector 不越界；
- stride 满足；
- tail 有单独处理。

## 7. 优化阶梯

建议顺序：

1. 正确的标量 baseline；
2. warp global access 合并；
3. block tile/shared reuse；
4. shared layout 与 bank conflict；
5. register/thread tile；
6. vectorized load/store；
7. 循环展开和 ILP；
8. double/multi-stage；
9. async copy；
10. Tensor Core；
11. split-K/Stream-K/persistent；
12. fusion/epilogue；
13. kernel family + dispatcher。

实际顺序由 profiling 决定，但每次只改变一个主要因素。

## 8. 编译器与生成代码

编译配置至少记录：

```text
target SM / PTX:
optimization:
fast math:
lineinfo/debug:
max register/launch bounds:
CUDA compiler version:
```

检查：

- `ptxas -v` 资源；
- PTX 是否出现目标语义；
- SASS 是否生成 vector load/MMA/async copy；
- 循环是否如预期展开；
- 是否有意外 local spill；
- source line 是否产生大量地址/转换指令。

源码写成 `float4`、WMMA 或 async API，不代表最终机器指令一定如预期。

## 9. Correctness 门禁

每个 backend 都应覆盖：

- 最小 shape；
- tile 边界前/等于/后；
- 非对齐 shape；
- 大 shape；
- current stream；
- 非默认 device；
- 错误 dtype/layout/device；
- 数值极端；
- 重复运行；
- sanitizer。

优化 backend 可以有 fast-path 约束，但公共 API 必须 dispatch/fallback。

## 10. Benchmark 门禁

必须避免：

- 未 warmup；
- 每次 iteration 强制全 device synchronize；
- 把一方 allocation/zero-fill 算入而另一方没有；
- 比较不同 math mode；
- 只取最小时间；
- profiler duration 直接替代正常运行；
- 只测一个甜点 shape；
- 忽略时钟/功耗波动。

保存 JSON/CSV，让结果可对比、可画 heatmap、可回归。

## 11. Profiling 门禁

固定顺序：

1. Nsight Systems 找真正热点；
2. 非 profiler event baseline；
3. NCU Summary/Launch/SOL；
4. Memory 或 Compute 深挖；
5. Scheduler/occupancy；
6. Source/SASS；
7. 形成一个可证伪假设；
8. 实现单一改动；
9. correctness + benchmark + profiler 对比。

详见 [Profiling 工作流](../profiling.md)。

## 12. 从一个 kernel 到生产实现

生产性能通常来自 kernel family：

```text
dispatch(shape, dtype, layout, arch, workspace, math_mode)
  ├── large-square kernel
  ├── skinny-M kernel
  ├── skinny-N kernel
  ├── small-K kernel
  ├── split-K / persistent kernel
  └── library/general fallback
```

还需要：

- feature/architecture guards；
- build/package；
- ABI 与错误信息；
- CUDA Graph；
- 多 GPU/stream；
- 数值与确定性契约；
- correctness/performance CI；
- 不支持分支的稳定 fallback。

## 13. Code review 清单

### 正确性

- 所有 global/shared 索引有边界证明；
- vector alignment 已验证；
- barrier 参与者一致；
- warp mask 正确；
- buffer reuse 前同步；
- 32/64-bit index 范围正确；
- stream/device 正确；
- launch error 可见。

### 性能

- warp lane 地址可推导；
- shared bank 映射可推导；
- 理论 global bytes 已计算；
- register/SMEM 资源已记录；
- grid waves 足够；
- 没有不必要 zero-fill/intermediate；
- 数学与地址指令比例合理；
- 改动在 shape 矩阵而非单点验证。

### 可维护性

- tile 常量集中；
- backend 名反映策略；
- fast path 与 fallback 分开；
- reference/test/benchmark 能独立运行；
- profiling 命令和结果可追溯。

## 14. 参考资料

- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [CUTLASS Efficient GEMM](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html)
- [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/)

