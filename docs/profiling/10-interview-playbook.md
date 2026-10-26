# 10. Kernel Profiling 面试与实战问答

## 1. “你如何分析一个未知 CUDA kernel？”

推荐回答：

1. 先确认 correctness、shape、dtype、layout、计时范围；
2. 用 NSYS 确认它是端到端热点，排除 launch/copy/sync；
3. 写 FLOP、最少 bytes、grid/wave 和资源静态模型；
4. 保存 event baseline；
5. NCU 看 Launch、SOL、Occupancy、Scheduler；
6. 根据方向进入 Memory 或 Compute；
7. 用 Source/SASS 定位指令；
8. 提出单因素改动；
9. correctness、duration、counter、shape 矩阵共同验证。

## 2. “Memory SOL 很高说明什么？”

只说明 memory 聚合子系统中至少有路径接近其持续吞吐模型。要继续区分 DRAM、L2、
L1/TEX、shared；不能直接说 DRAM bandwidth-bound。

## 3. “SOL 两边都低怎么办？”

查：

- grid 太小/wave tail；
- eligible warp 低；
- memory/execution dependency；
- barrier；
- divergence；
- block 负载不均；
- 指令 fetch/调度；
- 极短 kernel 与 launch overhead。

## 4. “Occupancy 只有 25%，要不要优化？”

先看目标管线是否已饱和、eligible/issued 是否充足、是否 spill。若 kernel 用大
register tile 获得高复用和 ILP，25% 可能足够。只有低 occupancy 导致 latency 无法
隐藏时才值得换取更多 resident warp。

## 5. “Long scoreboard 很高怎么处理？”

它表明 consumer 等待较长 memory dependency。定位 producer load，检查：

- 哪个 address space；
- sectors/coalescing；
- cache hit 与 DRAM traffic；
- 是否可复用到 shared/register；
- 是否能预取/流水；
- 是否有更多 ILP/warp；
- 是否是 spill。

## 6. “Shared-memory bank conflict 怎么验证？”

先推导 `word_address % 32`，再看 actual/ideal/excessive wavefront 与 source 指令。
修改 padding/layout后同时检查资源、occupancy 和 duration。

## 7. “Cache hit rate 低怎么办？”

先问数据是否应该复用。streaming 输入低 hit 正常。比较理论 bytes、request/sector、
miss 到下一级的流量和 duration；不要为 hit rate 人工保留没有复用价值的数据。

## 8. “如何判断 latency-bound 与 bandwidth-bound？”

- bandwidth-bound：对应层级 throughput 接近持续上限，减少 bytes 能改善；
- latency-bound：总 throughput 不高，但 eligible warp 少、scoreboard 等待，缺少
  TLP/ILP/MLP。

两者可能同时存在，且针对不同内存层级。

## 9. “为什么 profiler 下时间和平时不同？”

NCU 可能多 pass replay、保存恢复内存、patch 指令、改变 cache/频率环境；NSYS 也有
trace 开销。用正常 event benchmark 作为性能数字，profiler 用于解释。

## 10. “如何证明一个优化有效？”

必须同时满足：

- correctness；
- 非 profiler duration 稳定下降；
- 与假设一致的 counter/SASS 变化；
- 没有隐藏 allocation/preprocess；
- 多个目标 shape 成立；
- 明确退化范围和 fallback。

## 11. “为什么最高的 stall 不一定最该优化？”

stall 是 warp 状态采样，只有 issue slot 不足时才限制吞吐；它也常归因到 consumer。
若目标管线已经饱和，即使某 stall 占比高也可能不是主要优化方向。

## 12. “如何比较自定义 kernel 与 cuBLAS？”

统一：

- dtype/math mode；
- transpose/layout；
- alpha/beta；
- output allocation/initialization；
- warmup/cache；
- timing boundary；
- workspace；
- correctness；
- shape 集合。

cuBLAS/cuBLASLt 是 algorithm family/heuristic，不应只用一个 shape 宣布普遍胜负。

## 13. “为什么 NCU 指标可能超过 100%？”

百分比可能来自持续峰值模型、不同 pass、实例聚合、频率变化或异步 engine 干扰。
检查 metric 定义、suffix、replay pass 和测量稳定性，而不是先认为硬件超越理论峰值。

## 14. “什么是 request、sector、wavefront？”

- instruction 触发硬件工作；
- request 是提交给子系统的请求；
- sector 是常见的 32 B 对齐数据块；
- wavefront 是该子系统一次能处理的工作包；
- 一个 request 可涉及多个 sector，也可拆成多个 wavefront。

## 15. “为什么 source 行高亮 long scoreboard，但 load 在别处？”

stall sample 记录不能发射的 consumer 指令。producer load 更早发出但结果未返回，需
沿 SASS register dependency 回溯。

## 16. 白板诊断题

### 大方阵快，瘦长矩阵慢

优先考虑 tile 与 grid waves、wave tail、split-K、shape-specific tile。

### K 小时自定义 SGEMM 很差

考虑 launch、output memset、barrier、过大的 accumulator tile和固定开销。

### Register 从 120 降到 80，性能反而下降

可能 spill，或 compiler scheduling/ILP 变差；检查 local traffic、eligible warp、
pipeline 与 duration。

### Shared conflict 降了，性能不变

shared 不是瓶颈，或新增成本抵消；检查 SOL、scheduler、资源和其他 pipeline。

### Tensor Core utilization 高但整体慢

检查数据转换/layout、softmax/epilogue、barrier、load pipeline及 Tensor Core 阶段
占比。

## 17. 学习闭环

每完成一次 profiling，用五句话复盘：

1. workload 是什么；
2. 原瓶颈证据是什么；
3. 代码根因是什么；
4. 改动为何影响该根因；
5. 在什么范围内有效。

这比记住某个 NCU 百分比更能形成可迁移的 CUDA 能力。

