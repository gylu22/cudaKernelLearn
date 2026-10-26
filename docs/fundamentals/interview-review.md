# CUDA Kernel 面试复习提纲

本文不是标准答案列表，而是帮助把知识组织成“概念—原因—权衡—验证”的表达方式。
面试回答尽量包含一个反例或 profiling 方法，避免只背定义。

## 1. 推荐回答结构

面对性能问题：

1. 明确 workload、shape、dtype、layout 和计时范围；
2. 给出硬件/算法模型；
3. 说明可能瓶颈与权衡；
4. 说明用什么指标验证；
5. 给出不适用的情况。

例如“shared memory 一定更快吗”：

> shared memory 是低延迟、软件管理的片上存储，适合 block 内复用和重排。但它会
> 消耗每 block 资源，需要同步，还可能有 bank conflict。只有减少的 global traffic
> 大于新增成本，并在 profiler 中体现为 duration 下降，才是有效优化。

## 2. 执行模型

### Grid、block、warp、thread 的关系

- grid 是一次 launch；
- block 是 SM 调度与 shared-memory 协作边界；
- block 拆成 32-thread warp；
- scheduler 以 warp 为分组发射指令；
- thread 有独立索引、寄存器和 predicate。

延伸：普通 kernel 不保证 block 顺序；三维索引不自动改变内存布局。

### 什么是 SIMT？

同一 warp 的 lane 共享指令发射，但有独立数据和控制状态。分支不一致时各路径会在
不同 active mask 下执行。SIMT 不是传统 SIMD 寄存器接口，也不是 32 个完全独立
CPU thread。

### 什么是 divergence？

warp 内 lane 选择不同控制路径导致多条路径执行。成本取决于路径工作量与 active
lane；warp-uniform 分支不产生这种序列化。

## 3. 内存

### 解释 coalescing

一个 warp 的地址被合并为尽量少的对齐 sector。连续只是常见实现，核心指标是
useful bytes 与 physical sector/transaction 的关系。

### Register、local、shared 的区别

- register：thread 私有片上操作数；
- local：thread 私有地址空间，通常在 device memory 路径；
- shared：block 共享片上、软件管理；
- register 不够可能 spill 到 local；
- shared 受 bank 和 block 容量约束。

### Bank conflict 如何产生？

同一 warp 的 shared 指令访问同 bank 的不同地址，需要拆为多个 wavefront。同地址
读取可以广播。转置/padding 要按实际 word address `% 32` 推导并用指标验证。

### Cache hit rate 越高越好吗？

不一定。streaming workload 低 hit 正常；高 hit 也可能来自重复无效 load。先比较
理论最少 bytes、实际 sector 和最终 DRAM traffic。

## 4. 并行与资源

### 什么是 occupancy？

active warp 占架构最大 warp 的比例。它表示潜在 latency hiding，不是性能分数。
应继续看 eligible/issued warp、spill、复用和管线吞吐。

### Register 越少越好吗？

不是。更多寄存器可保留 accumulator、增加 ILP 和复用；过多则降低 residency 或
spill。最优点取决于 kernel。

### 如何选择 block size？

先选择 warp 整数倍候选，再结合工作映射、global/shared access、register/SMEM、
resident blocks、grid waves 和实测。不存在固定的 128/256 通用最优答案。

## 5. 同步

### Barrier 与 fence

barrier 协调参与线程到达；fence 约束内存观察顺序。`__syncthreads` 是 block
barrier，`__threadfence` 不是。

### Atomic 的性能问题

同地址竞争导致序列化；可先 thread/warp/block 局部聚合。atomic 的 scope 和顺序
也影响成本。浮点 atomic 顺序不定可能影响确定性。

### 为什么不能在普通 kernel 中写全局 barrier？

并非所有 block 同时驻留；已驻留 block 自旋可能占满 SM，使未调度 block 永远无法
到达 barrier。

## 6. 性能分析

### 如何判断 memory-bound？

需要某个内存层级吞吐接近持续上限，且减少对应流量或提高 AI 能降低 duration。
仅看到 long scoreboard 或低 cache hit 不足以下结论。

### Long scoreboard 是什么？

warp 的下一条消费者指令在等较长的 memory dependency，常与 global/local/L1TEX
路径相关。它是采样到的等待状态，需要回到 producer load、memory 指标和 source
定位。

### 为什么 SOL 两边都低？

可能是 grid 太小、依赖链、barrier、divergence、tail、不均衡或频繁 issue 空洞。
继续看 launch、scheduler、stall 与 source。

### Roofline 的局限

它给出吞吐上界和 AI 方向，不解释依赖、同步、发散、低并行度；FLOP/byte 口径和
数据类型路径不匹配时结论会错误。

## 7. GEMM

### GEMM 如何逐步优化？

```text
coalesced global access
→ shared block tiling
→ register/thread tiling
→ shared layout
→ vector load
→ software pipeline/async copy
→ Tensor Core
→ split-K / kernel family / dispatcher
```

### 为什么固定 GEMM kernel 只在部分 shape 快？

tile 对齐、grid waves、K 大小、长宽比和资源驻留随 shape 变化。库可以选择多个
algorithm，而单 kernel 只有一个配置。

### Split-K 的代价？

增加 K 方向并行，适合输出 block 少而 K 大；代价是 partial output、归约、workspace、
额外流量和可能的数值顺序变化。

## 8. FlashAttention

### 为什么比普通 attention 节省内存流量？

通过 Q/K/V tiling 与 online softmax，不把完整 N×N score/probability 写入 HBM；
保持行 max、normalizer 和输出 accumulator 并在新 tile 到来时重标定。

### Online softmax 为什么要重标定旧输出？

新 tile 可能提高全局 row max。旧指数项原本相对旧 max 计算，必须乘
`exp(old_max-new_max)`，normalizer 和输出 accumulator 都要缩放。

### FlashAttention 的瓶颈一定是 Tensor Core 吗？

不一定。softmax exp/reduction、寄存器 fragment、shared memory、barrier、Q/K/V
搬运、序列 tail 都可能限制性能。要分 QK、softmax、PV 与流水分析。

## 9. 数值

### 为什么 reduction 不确定？

浮点加法不满足结合律，block/atomic 到达顺序改变累加树。确定性算法需要固定顺序或
受控 reduction，常有性能代价。

### FP16、BF16、TF32 区别？

回答范围、fraction 精度、存储/计算语义和 accumulator，不只说“位数更少”。

## 10. 工程

### 一个 kernel 离生产还有什么？

- 一般 shape/tail；
- dtype/layout/stride/API；
- dispatcher/fallback；
- current stream、多 GPU、Graph；
- sanitizer 与随机测试；
- 数值/确定性契约；
- 架构兼容；
- benchmark/performance CI；
- build、package、错误信息。

### PTX 与 SASS 的区别？

PTX 是虚拟 ISA，可由工具链/JIT 映射到目标架构；SASS 是具体 GPU 机器指令。源码
意图是否实现最终要看目标架构 SASS。

## 11. 自测方式

对每个主题尝试在两分钟内回答：

1. 定义是什么；
2. 为什么会影响性能/正确性；
3. 一个反例；
4. 用什么指标或工具验证；
5. 在 SGEMM/FlashAttention 中的实例。

对应深入阅读：

- [GPU 硬件与执行模型](gpu-architecture.md)
- [GPU 内存层次](memory-hierarchy.md)
- [同步、Stream 与并发](execution-and-synchronization.md)
- [Kernel 性能模型](performance-model.md)
- [数值精度](numerical-precision.md)
- [Kernel 开发方法](kernel-development.md)
- [CUDA Profiling 工作流](../profiling.md)

