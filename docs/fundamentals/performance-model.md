# CUDA Kernel 性能模型

性能优化不是逐条套“最佳实践”，而是建立上界、测量现状、定位限制因素，再改变
最可能影响 duration 的一件事。本章给出 kernel 性能分析需要掌握的模型。

## 1. 先定义性能问题

同一个算子至少有三种不同问题：

| 问题 | 计时范围 | 典型优化 |
| --- | --- | --- |
| 单 kernel 为什么慢 | 已预分配数据上的 GPU kernel duration | tile、访存、流水、指令 |
| operator 为什么慢 | 分配 + dispatch + 一个或多个 kernel | fusion、allocator、dispatch |
| 应用为什么慢 | CPU、copy、通信、所有 operator | batching、并发、数据管线 |

必须先确定目标。把 kernel 优化 20%，若它只占应用 5%，按 Amdahl 定律端到端收益
仍然有限。

## 2. 延迟、吞吐、带宽

### 2.1 Latency

一项工作从开始到结果可用的时间。例如 memory load 的数据依赖 latency。

### 2.2 Throughput

单位时间可开始或完成多少项工作。流水化硬件可以让单项 latency 较高但整体
throughput 很高。

### 2.3 Bandwidth

单位时间跨某接口传输的字节数。要区分：

```text
theoretical peak bandwidth
sustained measured bandwidth
effective useful bandwidth
actual physical traffic bandwidth
```

优化目标通常是减少必要 physical traffic 或提高有效工作占比，而不是让总流量指标
越高越好。

## 3. 工作量模型

在 profiling 前先写出：

- 算术操作数量；
- 理论最少输入/输出字节；
- 中间数据量；
- 同步次数；
- grid/block 数；
- 每线程/warp/block 的任务；
- dtype 与指令路径。

SGEMM：

```text
FLOP ≈ 2MNK
minimum global bytes ≈ 4(MK + KN + MN)  // FP32，忽略 cache/重复读取
```

但 naive kernel 的 A/B 会被许多输出线程重复读取，实际流量远高于算法输入大小。

FlashAttention 的关键不是改变 `O(N²D)` 主要计算量，而是避免把完整 score 与
probability 矩阵写入/读回 HBM。

## 4. Arithmetic intensity 与 Roofline

```text
AI = operations / bytes_at_selected_level
attainable_performance <= min(compute_peak, bandwidth × AI)
```

同一 kernel 可以有多级 AI：

- DRAM AI；
- L2 AI；
- L1 AI；
- shared-to-register 的复用强度。

改变 tile 可能不改变算法 FLOP，却改变每一级物理 bytes。

### 4.1 Ridge point

```text
ridge_AI = compute_peak / bandwidth_peak
```

- AI 低于 ridge：更可能受该内存层级带宽约束；
- AI 高于 ridge：更可能受计算吞吐约束。

“更可能”不是最终诊断。依赖、同步、grid 太小或 pipeline 空洞会让点远离两条 roof。

## 5. TLP、ILP、MLP

### TLP

多个 resident/eligible warp 提供线程级并行。增加 block/warp 有助隐藏延迟，但受
寄存器、shared memory 和 grid 限制。

### ILP

同一 warp 有多条互不依赖的指令。thread tiling、多个 accumulator 和软件流水可
增加 ILP，但也增加寄存器。

### MLP

多个独立 memory request 同时在途。向量化、预取和多 stage 可提高 MLP；单条严格
依赖链无法靠同线程连续发 load 隐藏 latency。

常见失败：

```text
提高 occupancy 但复用下降 → DRAM 流量上升
增加 accumulator 但 spill → local 流量上升
增加 stage 但 SMEM 过大 → resident block 减少
循环展开过多 → code size、register 和 instruction cache 压力
```

## 6. Occupancy 模型

```text
occupancy = active warps / architecture maximum warps
```

它只表示驻留，不表示 ready。判断 occupancy 是否不足：

1. eligible warp 是否经常为零或很少；
2. issue slot 是否大量空闲；
3. 主要 stall 是否可由其他 warp 隐藏；
4. 目标 pipeline/带宽是否尚未饱和；
5. 增加 occupancy 的代价是什么。

低 occupancy 的 Tensor Core/GEMM kernel 仍可能很快，因为较大的 tile 提供高复用和
ILP。高 occupancy 的 naive kernel 也可能被 DRAM 流量压制。

## 7. Wave quantization 与负载均衡

设：

```text
G = grid block 数
S = SM 数
r = 每 SM resident blocks
slots_per_wave = S × r
```

粗略 wave utilization：

```text
G / (ceil(G / slots_per_wave) × slots_per_wave)
```

它没有包含 block duration 差异，但能解释固定 tile 在不同 shape 上的性能锯齿。

负载不均来源：

- tail block 工作更少或分支更多；
- 稀疏/不规则数据；
- early exit；
- 每个 block 处理的序列长度不同；
- persistent queue 分配不均。

## 8. Little's Law 的直觉

隐藏 latency 需要足够在途工作，可用直觉式理解：

```text
required_concurrency ≈ latency × desired_throughput
```

如果一次 global dependency 很长，而每个 warp 在等待前只发出一个请求，就需要很多
warp。若每个 warp 有多个独立请求/计算，ILP/MLP 可降低对极高 occupancy 的依赖。

## 9. 指令工作量与有效工作

总 instructions 少不一定快，更多 instructions 也不一定慢。要区分：

- necessary math；
- address/control overhead；
- predicated-off instructions；
- redundant recompute；
- data conversion；
- synchronization；
- spill；
- pipeline feed instructions。

score recompute 可能增加 FLOP，却因降低 shared/local 流量和寄存器压力而更快。

## 10. 优化决策树

```text
duration 是否可信？
  ├─ 否：修 benchmark
  └─ 是
      grid 是否填满 GPU？
        ├─ 否：更小 tile / split-K / batch / persistent
        └─ 是
            哪个资源接近 throughput 上限？
              ├─ DRAM/L2/L1/shared：减少流量或改善访问
              ├─ FP/Tensor/MUFU：减少工作或换指令路径
              └─ 都不高
                  eligible/issue 是否低？
                    ├─ memory dependency：复用、预取、MLP、更多 warp
                    ├─ execution dependency：ILP、展开、多个 accumulator
                    ├─ barrier：减少同步、平衡工作、pipeline
                    └─ tail/divergence：任务映射与调度
```

完整 profiler 流程见 [Profiling 工作流](../profiling.md)。

## 11. 性能声明应包含适用域

不要只写“比 cuBLAS 快”：

```text
GPU / CUDA / driver:
dtype / math mode:
shape/layout:
是否含 allocation:
warmup/repeats/statistic:
reference:
误差:
工作频率/功耗条件:
```

对 kernel family 应报告 shape heatmap、P50/P95，而不是只挑一个最佳点。

## 12. 面试问题

### Memory-bound 与 latency-bound 有什么区别？

memory-bound 通常指某内存层级吞吐接近可持续上限，更多带宽会改善性能；
latency-bound 指请求依赖等待且没有足够 TLP/ILP/MLP 隐藏，即使总带宽不高也会慢。

### 如何判断 kernel 是 compute-bound？

不能只看 FLOP 多。需要目标计算 pipeline 接近持续吞吐上限、scheduler 能稳定供给
指令、roofline 位于计算侧，并确认不是其他指令/同步限制。

### 为什么提高 arithmetic intensity 不一定更快？

新增复用可能增加 shared-memory 事务、barrier、寄存器、地址计算并降低 occupancy。
只有减少的瓶颈成本大于新增成本时 duration 才下降。

## 13. 参考资料

- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)
- [Nsight Compute Compute Triage](https://docs.nvidia.com/nsight-compute/ComputeTriage/)

