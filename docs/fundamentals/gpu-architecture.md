# NVIDIA GPU 硬件与 CUDA 执行模型

本文建立从 CUDA 源码到 GPU 硬件执行的完整映射。目标不是背诵某一代 GPU 的固定
核心数量，而是理解：一个 kernel 为什么能并行、资源在哪里分配、warp 为什么会
等待，以及这些现象如何反映到 profiler。

精确的 SM 数、每 SM 寄存器数量、shared-memory 容量、最大 resident warp/block
数量都依赖 compute capability。需要查询目标 GPU，而不是把某代架构的数据当成
CUDA 通用规则。

## 1. 从系统到执行单元

可以用下面的层次理解一块 NVIDIA GPU：

```text
Host / CPU
  └── CUDA context
      └── Device / GPU
          ├── copy engine、display/video 等非 SM engine
          ├── L2 cache、memory controller、HBM/GDDR
          └── 多个 GPC/TPC（具体组织因架构而异）
              └── SM (Streaming Multiprocessor)
                  ├── 多个 SM sub-partition
                  │   ├── warp scheduler / dispatch
                  │   ├── register file partition
                  │   └── FP/INT/LSU/SFU/Tensor 等执行管线
                  └── SM 范围共享的 L1/shared-memory 子系统
```

CUDA 编程最常直接接触的是 SM：

- thread block 被调度到某个 SM，并在完成前通常不会迁移；
- block 的寄存器和 shared memory 在进入 SM 时获得资源；
- block 被拆成 warp；
- warp 绑定到一个 SM sub-partition 的调度资源；
- warp scheduler 从 ready 的 warp 中选择下一条指令发射。

GPC、TPC、sub-partition 的数量和连接方式是微架构细节。写可移植 kernel 时不要依赖
它们的固定数量；profiling 时却要知道聚合指标可能掩盖不同 partition 的不均衡。

## 2. CUDA 软件层次与硬件层次

| CUDA 概念 | 主要硬件含义 | 重要性质 |
| --- | --- | --- |
| grid | 一次 kernel launch 的全部 block | 普通 launch 中 block 次序不保证 |
| block/CTA | 调度和资源分配单元 | 同一 block 驻留在同一 SM |
| warp | 通常 32 个连续线性 thread | scheduler 的基本执行分组 |
| thread/lane | 独立的寄存器、predicate、索引状态 | 同 warp lane 共享指令发射 |
| cluster | 支持架构上的 block 集群 | 可使用 cluster 同步与 distributed shared memory |

三维 `blockIdx`、`threadIdx` 主要服务索引表达。线程线性化顺序为 x 最快：

```text
linear_tid =
    threadIdx.x
  + blockDim.x * threadIdx.y
  + blockDim.x * blockDim.y * threadIdx.z
```

warp 由连续的 `linear_tid` 组成。因此，如果希望一个 warp 的 lane 访问连续内存，
通常应优先让连续数据映射到 `threadIdx.x`。

## 3. SM 内有哪些资源

### 3.1 Warp scheduler 与 dispatch

每个周期，scheduler 检查它管理的 resident warp：

- active/resident warp：已经获得 SM 资源，尚未结束；
- eligible warp：下一条指令已解码、操作数依赖满足，所需功能单元可接受工作；
- issued warp：当前周期实际发射了指令。

这是 profiling 中最重要的一条关系：

```text
resident 多，不代表 eligible 多；
eligible 多，不代表每个周期都能发射；
发射频繁，也不代表发射的是算法真正需要的有效工作。
```

warp 上下文长期保存在片上，因此硬件切换到另一 resident warp 不需要 CPU 线程
切换那样保存/恢复完整上下文。GPU 用大量 warp 和 ILP 隐藏流水线与访存延迟。

### 3.2 执行管线不是“CUDA Core 数”的简单总和

常见管线包括：

- FP32、FP64 浮点算术；
- INT/地址计算；
- load/store unit；
- special function unit，用于部分 reciprocal、sqrt、exp 等路径；
- Tensor Core，用于矩阵乘加类指令；
- branch、uniform/datapath 等架构相关单元。

同一条源码表达式可能生成多个 SASS 指令并占用不同管线。例如数组下标可能产生
整数乘加，`expf` 可能包含特殊函数与辅助指令，global load 会经过 LSU/L1TEX。

因此：

- “CUDA Core 利用率低”不是完整诊断；
- FP32 pipeline 高不代表 Tensor Core 高；
- Tensor Core 高不代表整个 SM 所有资源都饱和；
- 地址计算、数据类型转换和同步也可能成为瓶颈。

### 3.3 Register file

每个线程有自己的逻辑寄存器，但物理 register file 位于 SM/sub-partition。编译器
为 kernel 确定 registers/thread，硬件按分配粒度给 warp/block 分配寄存器。

影响包括：

- register tile 可以提高数据复用与 ILP；
- live range 过大提高 register pressure；
- 每线程寄存器增加可能降低 resident block/warp 数；
- 寄存器不足或动态索引数组可能进入 local memory；
- 手动限制寄存器可能提高 occupancy，也可能引入更昂贵的 spill。

`ptxas -v` 的 registers/thread 是起点，最终应结合 occupancy limit、local load/
store 和 duration。

### 3.4 Shared memory 与 L1

shared memory 是软件管理的 block 级片上存储。现代架构常让 shared memory 与 L1
data cache 使用统一或紧密相关的物理资源，但容量划分与配置依架构而异。

shared memory 的价值：

- block 内显式复用 global 数据；
- 在线程之间通信；
- 将 global 的不规则布局重排为计算友好布局；
- 为异步 global-to-shared pipeline 提供 stage buffer。

代价：

- 每 block 分配量限制 resident blocks；
- bank conflict 降低有效吞吐；
- barrier 与 producer/consumer 协议增加控制成本；
- tile 过大可能减少 grid 并行度。

### 3.5 L2、memory controller 与 device memory

L2 由全 GPU 的 SM 和部分 engine 共享，是 global/local/部分其他访问到 device
memory 前的重要缓存层。L2 后面连接 memory partition/controller，再到 HBM 或
GDDR。

需要区分：

- 峰值带宽：硬件规格中的理想上限；
- sustained bandwidth：特定访问模式和频率下可持续达到的带宽；
- effective bandwidth：算法有效字节数除以时间；
- physical traffic：实际穿过某一级的字节或 sector。

非合并访问可能让 physical traffic 很高，但 effective bandwidth 很低。

## 4. Block 如何驻留到 SM

一个 block 能进入 SM，需要同时满足：

```text
threads/SM 上限
warps/SM 上限
blocks/SM 上限
register file 容量与分配粒度
shared memory/SM 与 shared memory/block
架构对 cluster、barrier 等资源的限制
```

resident block 数由最先耗尽的资源决定：

```text
resident_blocks =
min(thread_limit,
    warp_limit,
    block_limit,
    register_limit,
    shared_memory_limit,
    other_arch_limits)
```

不能只用简单连续除法得到精确结果，因为寄存器和 shared memory 可能按固定粒度
round up。使用 Nsight Compute Occupancy Calculator 或 CUDA occupancy API。

### 4.1 Grid wave 与 tail effect

如果 GPU 有 `S` 个 SM，每 SM 对某 kernel 能驻留 `r` 个 block，那么一次理想
满 wave 约容纳：

```text
S × r blocks
```

网格 block 数不是这个数量的整数倍时，最后一轮会有空闲 slot。kernel block 很长、
网格又只有少数 wave 时，tail effect 很明显。

例如固定 128×128 输出 tile 的 SGEMM：

```text
grid_blocks = ceil(M/128) × ceil(N/128)
```

小矩阵或瘦长矩阵可能只有几十个 block，即使单 block 很快也无法使用整块 GPU。

## 5. SIMT、active mask 与分支

SIMT 表示多个 thread 以 warp 为执行分组共享指令发射，但每个 lane 有自己的数据、
predicate 和程序状态。

若条件在 warp 内不一致：

```cpp
if (predicate_for_each_lane) {
    path_a();
} else {
    path_b();
}
```

硬件需要执行被任意 lane 选择的路径，并屏蔽不参与的 lane。成本大致取决于：

- 实际执行了多少条路径；
- 每条路径工作量；
- active lane 比例；
- 编译器是分支还是 predication；
- 分支前后是否存在同步与负载不均。

不是所有 `if` 都昂贵：

- 条件对整个 warp 一致时没有 warp divergence；
- 很短的分支可能使用 predication；
- 边界分支只影响少数最后 warp 时总体成本可能很小。

### 5.1 Independent Thread Scheduling

Volta（compute capability 7.0）以后支持 independent thread scheduling。它提高了
发散线程调度灵活性，但不等于 32 个 lane 变成 32 个完全独立标量核心，也不表示
divergence 免费。

更重要的是：不能依赖“同 warp 天然 lockstep”实现跨 lane 内存通信。warp 内通过
shared/global memory 交换数据时，应使用正确的 `__syncwarp(mask)`；通过 shuffle
交换寄存器时，mask 必须准确描述参与 lane。

## 6. 延迟、吞吐与隐藏延迟

### 6.1 Latency

一条指令从发射到结果可被依赖指令使用需要时间。register FMA、shared load、L1/L2/
DRAM load 的 latency 量级不同，并会受 contention、cache hit、地址和架构影响。

### 6.2 Throughput

某管线每周期能接收/完成多少工作。即使单条指令 latency 较高，只要有足够多独立
指令和 warp，管线仍可保持高吞吐。

### 6.3 三类并行性

- TLP：多个 warp/thread 之间的并行；
- ILP：一个 warp 内相互独立的指令；
- MLP：多个独立内存请求同时在途。

典型权衡：

```text
更大 thread tile
  → 更多 accumulator 和 ILP
  → 更高寄存器复用
  → 也可能降低 occupancy

更小 tile
  → 更多 block 和 TLP
  → 但数据复用下降、访存增加
```

高性能 kernel 往往主动用寄存器换复用和 ILP，因此 occupancy 不一定高。

## 7. 从代码现象映射到 profiler

| 代码/结构 | profiler 中优先查看 |
| --- | --- |
| block 太少 | grid、waves、achieved occupancy、SM active |
| register tile 很大 | registers/thread、register occupancy limit、spill |
| shared tile | shared bytes/block、bank conflict、barrier |
| global stride | sectors/request、requested/actual bytes |
| 长依赖链 | eligible warps、wait/scoreboard、source line |
| warp 分支不一致 | branch efficiency、active threads/instruction |
| Tensor Core | tensor pipeline、MMA 指令、tensor roofline |
| `expf`/除法 | MUFU/相关管线、instruction mix、dependency |

profiling 的完整方法见 [CUDA Profiling 工作流](../profiling.md)。

## 8. 常见面试问题

### GPU 为什么适合高吞吐任务？

GPU 使用大量执行资源与硬件多线程，通过 TLP/ILP/MLP 隐藏延迟，适合可以分解出大量
相似并行工作的任务。它不是单线程 latency 优先设计；工作量小、依赖串行或频繁
host/device 往返时未必有优势。

### Block 和 warp 的区别？

block 是编程与资源分配边界，可共享 shared memory 并进行 block barrier；warp 是
SM scheduler 的基本执行分组。一个 block 包含一个或多个 warp。

### Occupancy 为什么不是越高越好？

occupancy 只说明 resident warp 比例。若已有足够 eligible warp 饱和目标管线，
进一步提高 occupancy 没有收益；若通过缩小 tile/寄存器实现，还可能破坏复用或产生
spill。

### CUDA Core、SM、warp scheduler 是什么关系？

SM 包含调度、寄存器、片上存储和多类执行管线。“CUDA Core”通常指部分标量算术
执行资源，不是独立调度单元。scheduler 发射 warp 指令到适合的管线。

## 9. 参考资料

- [CUDA Programming Guide: Advanced Kernel Programming](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html)
- [CUDA Programming Guide: Writing SIMT Kernels](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html)
- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)
- [CUDA GPU Compute Capability](https://developer.nvidia.com/cuda-gpus)

