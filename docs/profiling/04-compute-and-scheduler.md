# 4. Compute、Scheduler 与 Warp Stall

![Pipeline utilization 示例](assets/pipeline-utilization.png)

Compute Workload Analysis 展示各执行管线承受的指令吞吐。柱子高表示该管线繁忙，
但“最高的柱子”不自动等于根因；先确认 scheduler 是否无法持续 issue。

## 执行管线

- FP32/FP64/Tensor：对应浮点或 MMA 指令；
- INT/ALU：整数算术、部分地址计算和控制相关指令；
- LSU/L1TEX/MIO：load-store、shared memory、特殊数学或相关队列，具体分类依架构；
- MUFU：倒数、指数、三角函数等特殊函数路径。

`--use_fast_math` 主要改变除法、平方根和部分特殊函数的精度/实现，不是让普通
FP32 FMA 普遍变快的开关。必须单独做数值与性能实验。

## 先看 Scheduler Statistics

重点是每 scheduler 每周期：

- active warps：已驻留；
- eligible warps：依赖已满足、可 issue；
- issued warps：实际发射。

active 很多但 eligible 很少，说明 occupancy 存在却没有隐藏住延迟。eligible 很多
但 issued 受限，则更像执行管线或发射带宽限制。

## Stall reason 的正确解释

![Warp stall 示例](assets/warp-stall.png)

- **long scoreboard**：等待 L1TEX 相关的较长延迟依赖，常见于 global/local
  memory；需结合 memory section。
- **short scoreboard**：等待 MIO 相关依赖，常见于 shared memory，也可能来自
  特殊数学或动态分支。
- **barrier**：等待 block/warp 同步；可能是同步太频繁，也可能是工作不均衡。
- **wait**：固定延迟执行依赖；增加独立指令、展开或更多可用 warp 可能帮助。
- **not selected**：warp 已 eligible，但本周期选择了另一个 warp；通常不是坏事。
- **MIO/LG throttle**：对应指令队列压力，需减少请求、提高访问宽度或交错计算。

Stall 图是“每条 issued instruction 之间 warp 处于各状态的平均周期”或采样分类，
不是各原因对 wall time 的直接百分比分摊。只有 scheduler issue 不足时才优先处理
stall，并用 duration 验证改善。

## 分支发散

warp 内 lane 走不同控制流时，路径通常被串行执行或 predication。排查 active
threads per instruction、branch efficiency 和 Source/SASS。

不要机械地用三元表达式或位运算替代 `if`：编译器可能生成相同 predication，也
可能执行更多无用指令。优先让同一 warp 的任务更一致，并比较生成指令与 duration。

## Active、Eligible、Issued 的诊断矩阵

| Active | Eligible | Issued | 解释 |
| --- | --- | --- | --- |
| 低 | 低 | 低 | residency/grid 不足，或最后 wave |
| 高 | 低 | 低 | warp 多但都在等依赖/同步 |
| 高 | 高 | 高 | scheduler 有充足工作，继续看管线吞吐 |
| 高 | 高 | 中/低 | dispatch/执行资源竞争或指标口径需下钻 |

“issued 高”也要确认 active lanes 与有效工作；大量 predicated-off instruction 仍可
让发射看起来繁忙。

## 依赖链与 ILP

```cpp
sum = fma(a0, b0, sum);
sum = fma(a1, b1, sum);
sum = fma(a2, b2, sum);
```

只有一个 accumulator 时存在连续 RAW dependency。使用多个独立 accumulator：

```cpp
sum0 = fma(a0, b0, sum0);
sum1 = fma(a1, b1, sum1);
```

可增加 ILP，但增加 register。GEMM thread tile 正是用多个 accumulator 换取复用与
ILP；最终还需合并或写回。

## Barrier stall 的拆解

高 barrier 可能来自：

- K-loop 每个 tile 两次全 block barrier；
- warp 工作量不同；
- tail lane 做额外分支；
- producer load 不均；
- barrier 前长依赖；
- 不必要使用 block barrier 而非 warp 协作。

不能直接删除 barrier。先画 producer/consumer buffer 生命周期，证明哪些参与者需要
同步，再缩小 scope 或流水化。

## 特殊函数

FlashAttention softmax 的 `expf`、reciprocal 等可能使用 MUFU 和辅助指令。检查：

- 调用次数是否可减少；
- 是否由少量 lane 串行执行；
- 近似数学是否允许；
- reduction 与 exp 能否并行；
- Tensor Core 加速 QK/PV 后，MUFU 是否成为新瓶颈。

## 参考资料

- [Nsight Compute Profiling Guide: Warp Scheduler States](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)
- [CUDA Best Practices Guide: Control Flow](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
