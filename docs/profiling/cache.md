# GPU Cache：从流量和复用解释命中率

缓存分析不能只问“hit rate 高不高”，应先问该数据是否本来就会复用、一次逻辑请求
产生了多少 sector、未命中最终增加了多少 DRAM 流量。

## L1/TEX

现代 NVIDIA 架构常把 L1 data cache、texture path 与 shared-memory 相关资源放在
统一或紧密关联的子系统中，具体行为随架构和 load cache operator 改变。

常见低效来源：

- warp 地址分散，单请求覆盖过多 sector；
- 工作集超过有效容量且有复用需求；
- local-memory spill 污染/占用缓存路径；
- 多个数据流相互驱逐；
- cache policy 与 streaming/reuse 模式不匹配。

非合并访问主要表现为“传输很多无效 sector”，不必然表现为低 hit rate；即使命中
L1，过多 sector 仍会占用 L1TEX 吞吐。

## L2

L2 为全 GPU 共享。一次性 streaming 输入的 L2 hit rate 低是预期现象，不等于
程序有 bug。L2 对跨 block/跨 kernel 的时间局部性、读写合并和 DRAM 流量更关键。

评估时比较：

- L2 read/write sectors；
- DRAM read/write bytes；
- 相同输入重复 launch 时 cold/warm cache 的差异；
- 并发 kernel 是否改变工作集和驱逐。

## DRAM 与 Unified Memory

DRAM 没有通常意义上的“cache miss”。L2 未命中会访问显存；Unified Memory page
fault/migration 是另一层地址驻留问题，应使用相应 UM 指标和 Nsight Systems 分析。

## Local memory

local memory 是线程私有的地址空间，但物理存储通常在 device memory，并走缓存。
数组动态索引、大线程私有对象或 register spill 都可能生成 local load/store。应
结合 `ptxas -v`、SASS 和 NCU local sectors 判断。

## 实验方法

1. 写下理论最小读写字节数。
2. 对比请求数、sector 数、L2/DRAM bytes。
3. 改变问题规模，观察跨过 L1/L2 容量时的拐点。
4. 用 baseline 比较 duration，而不是追求单一 hit-rate 数字。

## Cold、Warm 与容量实验

设计三类实验：

1. cold：每次使用新数据或显式改变工作集；
2. warm：重复相同输入；
3. capacity sweep：逐渐扩大工作集，跨过可能的 L1/L2 有效容量。

如果只有 warm 情况快，结论应说明依赖 cache residency，不能假定真实应用也满足。

## Cache 与合并访问是两个维度

```text
地址分散但数据命中 L1
→ hit rate 高，仍可能有大量 sector/wavefront

地址完全合并但一次性 streaming
→ hit rate 低，访问仍可能接近理想
```

因此先看 sectors/request 与有效字节，再解释 hit rate。

## Cache 优化的代价

主动提高复用可能需要：

- 更大 tile；
- block swizzle；
- persisting L2 window；
- shared staging；
- fusion。

这些方法会改变并行度、资源或调度顺序。必须在真实并发和工作集下验证，尤其注意
多个 kernel/stream 对 L2 的竞争。

## 参考资料

- [CUDA Best Practices Guide: Device Memory Spaces](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Nsight Compute Profiling Guide: Memory Workload Analysis](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)
