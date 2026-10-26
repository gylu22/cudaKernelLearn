# SGEMM 优化思路与性能差距分析

本文针对仓库当前的 FP32 SGEMM 实现，回答两个问题：

1. 现有 kernel 是沿着什么思路逐步优化的；
2. 为什么最好的 kernel 在部分矩阵上能达到 cuBLAS 的 95% 以上，在另一些矩阵上却有明显差距。

这里讨论的是当前仓库的真实实现，而不是一份脱离代码的通用 GEMM 教程。后续生产化工作见 [SGEMM 生产化差距与路线](production-roadmap.md)。

## 1. 先建立性能模型

SGEMM 计算：

```text
C[M, N] = A[M, K] × B[K, N]
```

总计算量约为：

```text
2 × M × N × K FLOP
```

优化的核心不是单纯减少指令，而是让数据在寄存器、共享内存和全局内存之间被充分复用，同时维持足够的并行度和指令级并行。

对当前 `128 × 128 × 8` 的 block tile：

- 每个 K tile 从全局内存读取 `128 × 8` 个 A 元素和 `8 × 128` 个 B 元素；
- 输入流量为 `(128 × 8 + 8 × 128) × 4 = 8192` 字节；
- 计算量为 `2 × 128 × 128 × 8 = 262144` FLOP；
- 忽略 C 的读写后，理想的输入算术强度约为 `32 FLOP/Byte`。

这解释了为什么 block tiling 很重要：同一份 A、B 数据被 block 内多个线程重复使用后，kernel 才可能从全局内存受限逐步转向计算流水线受限。

## 2. 当前 kernel 的优化链路

实现位于 `csrc/sgemm/kernels/simt/sgemm_fp32.cu`。

| backend | 主要方法 | 获得的收益 | 当前限制 |
| --- | --- | --- | --- |
| `naive` | 一个线程计算一个 C 元素 | 正确性基线、支持一般 shape | A/B 几乎没有跨线程复用 |
| `smem_tiled` | `32 × 32 × 32` 共享内存分块 | 减少全局内存重复读取 | 1024 线程/block；每线程只算一个元素，寄存器复用弱 |
| `thread_tile` | `128 × 128 × 8` block tile；每线程计算 `8 × 8` | 增强寄存器复用和 ILP | 固定 tile；共享内存访问仍有冲突 |
| `transposed` | A 在共享内存中转置 | 改善 A 的 shared-memory 访存模式 | B 的访问和固定 tile 问题仍在 |
| `double_buffered` | 两组 shared-memory buffer | 为加载与计算重叠建立结构 | 当前仍是同步加载，没有真正异步流水 |
| `padded` | 为 B 的 shared-memory 行增加 padding | 缓解部分 bank conflict | 固定 shape 约束和同步流水仍存在 |

最优的几个 backend 使用：

- block tile：`BM = BN = 128`；
- K tile：`BK = 8`；
- thread tile：`TM = TN = 8`；
- block：`16 × 16 = 256` 个线程；
- 每线程：64 个 FP32 accumulator；
- 向量化的 `float4` 全局内存访问。

这是一个典型的 SIMT GEMM 结构：全局内存负责搬运 tile，共享内存负责 block 内复用，寄存器负责线程内复用。

## 3. 为什么某些 shape 能达到 cuBLAS 95%+

达到 95% 以上并不反常，它说明当前固定配置恰好覆盖了一块“甜点区”。

### 3.1 M、N 与固定 tile 高度匹配

当 M、N 都是 128 的倍数时：

- 不需要边界判断和 predication；
- 每个 block 都能完成完整的 `128 × 128` 输出；
- 向量化 load/store 容易保持对齐；
- 不会因边缘 tile 浪费线程和计算。

当前优化 backend 直接要求 M、N 能被 128 整除，因此跑到的本来就是对它最友好的问题集合。

### 3.2 大型方阵能提供足够多的 block

输出网格中的 block 数为：

```text
G = ceil(M / 128) × ceil(N / 128)
```

当前机器是 30 个 SM。以方阵为例：

| shape | block 数 | 对全 GPU 的含义 |
| --- | ---: | --- |
| `128 × 128` | 1 | 只有一个 SM 有工作 |
| `1024 × 1024` | 64 | 只有很少几轮 wave，尾部 wave 可能不满 |
| `4096 × 4096` | 1024 | 有足够多的 block 摊平调度和尾部损失 |

若每个 SM 可同时驻留 `r` 个该 kernel 的 block，可用下面的粗略指标观察 wave 尾部：

```text
wave_efficiency ≈ G / (ceil(G / (SM_count × r)) × SM_count × r)
```

`r` 不能只靠静态猜测，应从 Nsight Compute 的实际 occupancy、寄存器和 shared-memory 限制确认。

### 3.3 大 K 能摊薄固定开销

K 较大时：

- 64 个 accumulator 被持续复用；
- kernel launch、输出初始化、首尾同步等固定成本占比下降；
- thread tile 带来的指令级并行更容易隐藏 FMA latency；
- A/B tile 的数据复用更充分。

因此，“大、方、对齐、K 足够大”正是当前 kernel 最有机会接近 cuBLAS 的区域。

### 3.4 比较精度目前是相对公平的

benchmark 已关闭：

```python
torch.backends.cuda.matmul.allow_tf32 = False
```

因此当前 FP32 SIMT kernel 没有被拿去和默认允许 TF32 Tensor Core 的 `torch.matmul` 比较。这个设置应继续保留并明确记录。

## 4. 为什么另一些 shape 会拉开差距

核心原因是：当前实现是一套固定的 micro-kernel，而 cuBLAS/cuBLASLt 可以根据问题 shape、硬件和可用 workspace 从多种算法配置中选择。95% 是某个工作区间的结论，不等同于全 shape 性能。

### 4.1 小矩阵：并行度不足，固定成本占主导

`128 × 128` 只有一个 block。即使这个 block 内部效率很高，其余 SM 仍然空闲。

小矩阵还更容易被以下成本主导：

- kernel launch；
- PyTorch 输出 tensor 创建；
- 当前 binding 使用 `torch::zeros` 初始化输出；
- 同步与循环控制。

尤其需要注意：当前 padded kernel 的 epilogue 形式保留了 C 的读取，即使 `beta=0`；因此 binding 为所有自定义 backend 分配了全零输出。该初始化会产生额外的 `4 × M × N` 字节写流量，而比较对象不一定承担完全相同的工作。K 越小，这个误差占比越明显。

### 4.2 瘦长矩阵：输出网格无法填满 GPU

例如 `M = 4096, N = 128` 时只有：

```text
32 × 1 = 32 blocks
```

在 30 个 SM 上，这只是接近一轮 block。若资源限制使每个 SM 只能驻留一个 block，就几乎没有额外并行度用于隐藏长延迟；即使能驻留两个，第二个驻留槽也基本用不上。

对此更合适的做法可能是：

- 使用 `128 × 64`、`64 × 128` 或 `64 × 64` tile；
- 对 K 很大但输出 tile 数很少的问题使用 split-K 或 Stream-K；
- 为不同长宽比分派不同 kernel。

### 4.3 小 K：同步、零初始化和循环开销难以摊薄

当前 `BK = 8`。K 很小时：

- 总 FLOP 少；
- 每个输出仍需占用 64 个 accumulator；
- barrier、地址计算、输出写回和输出清零占比上升；
- 大 `128 × 128` tile 可能牺牲 occupancy，却没有足够计算来换回收益。

小 K 通常需要更小的 tile、专用 kernel，或不同的向量化方向。

### 4.4 很大 K：同步 BK=8 流水暴露瓶颈

K 很大时，循环次数为 `K / 8`。当前 double-buffered 版本虽然使用两组 shared-memory buffer，但全局读取仍然是普通同步 load，没有真正通过 `cp.async` 将下一 tile 的搬运与当前 tile 的计算重叠。

这会带来：

- 更多 K-loop 迭代和 barrier；
- global-to-shared latency 暴露；
- 较小 BK 带来的地址计算和循环控制开销；
- 无法像多 stage pipeline 那样稳定向计算流水线供数。

应实际测试 `BK = 16/32` 和 2–4 stage pipeline，而不是默认 BK 越大越好。BK 增大会增加 shared-memory 使用，也可能降低 occupancy。

### 4.5 固定寄存器块会限制 occupancy

每线程 64 个 FP32 accumulator 已经至少占用 64 个寄存器，尚未计入 A/B fragment、地址、循环变量和编译器临时值。256 线程/block 会快速消耗每个 SM 的寄存器文件。

结果可能是：

- 每个 SM 的 resident block 数下降；
- 小网格更难隐藏延迟；
- 编译器调度或额外临时变量稍有变化，就可能造成明显性能变化。

必须结合 Nsight Compute 的 `launch__occupancy_limit_registers`、实际 active warps 和 local-memory spill 判断，而不能只看理论 occupancy。

### 4.6 shared-memory bank conflict 尚未完全消除

`transposed` 版本主要改善 A 的 shared-memory 布局，`padded` 版本进一步调整 B 的行跨度；但每种布局是否有效与 warp 内线程到地址的映射有关。

应检查：

- shared load/store bank conflict 指标；
- 每条 shared-memory 指令的 transactions/request；
- padding 是否只是把冲突从一类访问移到了另一类访问；
- 新布局是否增加了地址计算或 shared-memory 占用。

不要把“加 padding”当作自动成立的优化，需要用指标和对照实验验证。

### 4.7 cuBLAS 的比较对象不是单一固定 kernel

cuBLAS/cuBLASLt 提供算法与 heuristic 选择能力。对不同 shape，它可以选择不同的 tile、分块方式和实现路径。具体内部策略不公开，因此不应把某一次性能差距简单归因为某个不可见实现；更可靠的结论是：

- 当前自定义 kernel 只覆盖一个固定配置；
- 库有更大的候选配置空间；
- shape 改变后，双方使用的有效并行策略可能完全不同。

## 5. 如何定位某个 shape 的差距

先按问题类型分类，再看指标，避免只对单个结果猜原因。

| shape 特征 | 优先怀疑 | 首要检查 |
| --- | --- | --- |
| 小 M、小 N | GPU 未填满、launch/分配开销 | block 数、GPU duration、端到端时间 |
| M 或 N 很窄 | wave 尾部、tile 不匹配 | grid、waves、SM active cycles |
| K 很小 | 固定开销、输出清零、tile 过大 | 各阶段时间、DRAM bytes、barrier |
| 大型方阵 | 计算流水线、bank conflict | FLOP throughput、shared transactions、stall reason |
| K 很大 | global-to-shared 流水不足 | memory latency、barrier、eligible warps |
| 非 128 倍数 | 当前快 kernel 不支持 | tail 策略和 fallback 成本 |

建议至少记录：

- `M/N/K`、dtype、布局、是否允许 TF32；
- 自定义 kernel 与 cuBLAS 的中位数和 P95 latency；
- TFLOP/s 与 `custom / cuBLAS` 比值；
- block 数和估算 waves；
- registers/thread、shared memory/block、achieved occupancy；
- DRAM throughput、L2 hit rate；
- shared-memory bank conflicts；
- warp stall reason；
- 是否包含分配和输出初始化。

## 6. 推荐的 shape 矩阵

不要只测多个方阵。至少覆盖：

```text
方阵：   128, 256, 512, 1024, 2048, 4096
宽矩阵： M=128/256, N=4096, K=128/1024/4096
高矩阵： M=4096, N=128/256, K=128/1024/4096
小 K：   K=8/16/32/64
大 K：   K=1024/4096/8192
边界：   M/N/K 在 tile 边界前后，例如 127/128/129
```

将 `custom / cuBLAS` 画成以 M、N、K 分类的热力图，比单一平均值更能说明 kernel 的有效工作区间。

## 7. 推荐的下一轮优化顺序

1. 先修正 benchmark 公平性：移除 `beta=0` 路径不必要的 C 读取和 `torch::zeros`，区分纯 kernel 时间与包含分配的端到端时间。
2. 建立 shape sweep，给当前每个 backend 画性能区域，而不是只报最佳点。
3. 以 `transposed` 或经过验证的 `padded` 版本作为干净基线，复核 shared-memory conflict 和寄存器占用。
4. 比较 `BK=8/16/32`，同时记录 occupancy、barrier 和 global-load latency。
5. 实现真正的 Ampere `cp.async` 多 stage pipeline。
6. 增加多组 block/thread tile，为方阵、窄 N、窄 M、小 K 分别服务。
7. 对 block 数不足而 K 很大的 shape 实现 split-K 或 Stream-K。
8. 增加 FP16/BF16/TF32 Tensor Core 路径，并明确累加与数值误差契约。
9. 最后建立 dispatcher，用 shape、dtype、架构和实测阈值选择 kernel；不适用时回退到 cuBLAS。

生产系统的目标不是让单个 kernel 在所有 shape 上获胜，而是让整个 dispatcher 在目标 workload 上稳定、正确，并且没有明显性能悬崖。

## 8. 参考资料

- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)
- [CUTLASS Efficient GEMM](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html)
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html)
