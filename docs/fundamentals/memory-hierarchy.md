# GPU 内存层次、访问路径与优化方法

写 CUDA kernel 时，数据“放在哪里”和“以什么访问模式移动”通常比算术表达式本身
更重要。本章从程序员可见空间、硬件 cache、访存事务和常见优化模式四个层次建立
完整模型。

不要给每一级内存背固定 latency。latency、容量、带宽和 cache 行为随架构、频率、
访问模式及并发压力变化。更稳定的知识是可见范围、管理者、共享范围、事务规则和
性能风险。

## 1. 总览

| 层次/空间 | 管理方式 | 可见/共享范围 | 主要优点 | 主要风险 |
| --- | --- | --- | --- | --- |
| register | 编译器 + 硬件 | thread 私有 | 最高吞吐、用于操作数 | 容量有限、pressure、spill |
| local memory | 编译器 | thread 私有地址空间 | 可寻址私有存储 | 物理上在 device memory，延迟高 |
| shared memory | 程序员 | block；特定架构可到 cluster | 可控复用与通信 | bank conflict、barrier、occupancy |
| L1 data/TEX | 硬件 | 通常 SM 范围 | 缓存 global/local 等请求 | 容量有限、路径共享、架构相关 |
| constant cache | 硬件 | device 只读空间的 cache | warp uniform 访问很高效 | warp 多地址会序列化 |
| texture/read-only | 硬件 | device 只读/纹理语义 | 空间局部性与特殊寻址 | 不适合所有普通线性访问 |
| L2 | 硬件 | 全 GPU 共享 | 跨 SM、跨 kernel 复用 | 容量竞争、全芯片流量 |
| global/device memory | 程序员分配 | device/context | 大容量、持久 | 高 latency、带宽有限 |
| pinned host memory | host/runtime | CPU 与 DMA | 异步传输、DMA | 锁页资源有限 |
| mapped/zero-copy | runtime | GPU 可直接访问 host | 避免显式 copy | PCIe/互连 latency 与带宽 |
| Unified Memory | runtime/driver | 统一虚拟地址 | 易用、可迁移 | page fault、migration、抖动 |

## 2. Register

### 2.1 编译器为什么使用寄存器

算术指令通常直接读取和写入寄存器。线程局部标量、循环变量、指针、fragment 和
accumulator 都可能占用寄存器。

寄存器使用的三个重要概念：

- live value：当前值以后还会被使用；
- live range：值从定义到最后一次使用的区间；
- register pressure：同一时刻需要保留的 live value 太多。

代码中声明 100 个变量不代表同时需要 100 个寄存器；反过来，一个展开循环或大型
数组也可能让很多值同时存活。

### 2.2 Register pressure 的后果

1. registers/thread 上升；
2. 每 block 需要的寄存器增加；
3. resident block/warp 可能下降；
4. 编译器可能把值放到 local memory；
5. spill 增加 global/cache 路径流量。

优化方式不是盲目降低寄存器，而是缩短不必要 live range、减少同时存在的 fragment、
调整展开和 tile，在复用/ILP 与 occupancy 间做实验。

### 2.3 如何检查

```bash
nvcc ... --ptxas-options=-v
```

关注：

- registers；
- spill stores / spill loads；
- stack frame；
- NCU 中 local load/store 与 source/SASS 的 `LDL`/`STL`；
- register occupancy limit。

`--maxrregcount` 和 `__launch_bounds__` 是实验/约束工具，不是自动优化开关。

## 3. Local memory

local memory 是每线程私有的逻辑地址空间，但通常物理存储在 device memory，并经
L1/L2 cache。常见来源：

- register spill；
- 编译器无法静态索引的线程私有数组；
- 过大的线程局部对象；
- ABI、函数调用或栈需求。

例子：

```cpp
float values[64];
float x = values[runtime_index];
```

如果编译器不能把动态索引转换为可选择的寄存器访问，数组可能落入 local memory。

“源码中没有 local 指针”不代表没有 local traffic。必须看编译结果。

## 4. Shared memory

### 4.1 为什么 shared memory 能提速

它可以把一次 global load 转化为 block 内多次复用：

```text
global A/B tile
    ↓ cooperative load
shared A/B tile
    ↓ many reuse
register fragment
    ↓ FMA
```

也可以用来转置/重排，使 global load 与 compute 阶段分别采用各自最优布局。

### 4.2 静态与动态 shared memory

```cpp
__shared__ float tile[128];

extern __shared__ float dynamic_buffer[];
kernel<<<grid, block, bytes, stream>>>(...);
```

两者都会计入每 block shared-memory 资源。动态 shared memory 较大时，某些架构还
需要设置相应 kernel attribute 或 carveout。

### 4.3 Bank 模型

对 compute capability 5.x 及之后的常见模型：

- 32 个 bank；
- 连续 32-bit word 映射到连续 bank；
- 可用 `bank = word_address % 32` 做第一步推导；
- 一条 warp shared 指令访问不同 bank 可并行；
- 同地址读取可 broadcast；
- 同 bank 不同地址需要拆为多个 wavefront。

示例：

```cpp
__shared__ float tile[32][32];

// 固定 row，lane 访问连续 column：通常映射到不同 bank
x = tile[row][lane];

// 固定 column，lane 访问连续 row：stride=32，容易映射到同一 bank
x = tile[lane][column];
```

padding：

```cpp
__shared__ float tile[32][33];
```

把行跨度从 32 改成 33，使 `tile[lane][column]` 的 bank 随 lane 变化。但必须对
实际类型、向量宽度、访问指令和布局重新推导；padding 也会增加容量和地址计算。

### 4.4 Vector access 与 wavefront

每 lane 读取 64/128 bit 时，一条 warp 指令覆盖的数据可能超过一个 bank cycle，
即使没有传统意义上的同地址冲突，也可能需要多个 wavefront。应同时看：

- logical request；
- sectors；
- shared wavefront；
- ideal wavefront 与 excessive wavefront；
- 实际 source/SASS 指令宽度。

### 4.5 同步

shared memory 是通信媒介，不自动建立 happens-before：

```cpp
shared[tid] = value;
__syncthreads();
value = shared[other_tid];
```

如果只在一个 warp 内通信，也要基于参与 mask 使用 shuffle 或 `__syncwarp()`。
Volta+ 不应依赖隐式 lockstep。

## 5. Global memory 与合并访问

### 5.1 从 lane 地址到 sector

一次 warp memory instruction 包含所有 active lane 的地址。硬件将这些地址组合为
request，并访问一个或多个 sector。NCU 常用的模型中：

- sector 是对齐的 32-byte 数据块；
- L1/L2 cache line 常由 4 个 sector 组成，即 128 byte；
- 一个逻辑指令可能生成多个 request、sector 和 wavefront。

32 个 lane 各读一个连续 FP32：

```text
32 lanes × 4 B = 128 B useful data
```

若起始地址对齐且 lane 连续，通常覆盖 4 个 32 B sector。若每 lane stride 很大，
可能覆盖多达 32 个 sector，物理流量远大于有效数据。

### 5.2 合并访问的本质

目标是让同一 warp 的 active lane 地址覆盖尽量少的对齐 sector：

```cpp
// 通常友好
out[base + lane] = in[base + lane];

// stride 访问可能低效
out[lane * stride] = in[lane * stride];
```

需要同时考虑：

- 首地址对齐；
- 每 lane 访问宽度；
- active mask；
- 跨 cache-line 边界；
- tail；
- load/store 方向；
- cache 是否复用了额外 sector。

### 5.3 `float4` 能做什么、不能做什么

它可能：

- 减少 load/store 指令数量；
- 增大单线程访问宽度；
- 帮助生成 128-bit instruction。

它不能：

- 自动修复 warp lane 间的 stride；
- 取消自然对齐要求；
- 自动处理 tail；
- 保证更少 physical sector；
- 保证更快，尤其在 register pressure 增加时。

### 5.4 Effective bandwidth

对纯 copy：

```text
effective_bandwidth =
    (useful_read_bytes + useful_write_bytes) / time
```

若 profiler 显示 DRAM 实际流量明显更大，应检查非合并访问、重复加载、write
allocate/cache 行为、spill 和中间结果。

## 6. L1/TEX

现代架构的 L1 data cache、texture 路径与 shared-memory 子系统关系紧密，但具体
统一方式和容量配置随架构变化。

分析 L1 时不要只看 hit rate：

- 高 hit rate + 大量重复无效请求仍可能很慢；
- streaming 数据低 hit rate 可能完全正常；
- 非合并访问即使命中 L1，也会占用过多 sector/wavefront；
- local spill 也使用相关路径；
- shared-memory 指令可能反映到 L1TEX/MIO 相关指标。

优先顺序：

1. 算法最少需要多少字节；
2. 实际 request/sector/wavefront；
3. 哪些流量 hit/miss；
4. 哪一级吞吐接近上限；
5. source line 是谁产生流量。

## 7. L2

L2 为整个 GPU 共享，适合：

- 不同 SM 对相同数据的复用；
- 连续 kernel 间的时间局部性；
- 吸收部分重复/错位 sector；
- 减少到 device memory 的流量。

但 L2 hit rate 受以下因素影响：

- 工作集与 L2 容量；
- block 调度顺序；
- 多 kernel/多进程竞争；
- cold/warm cache；
- persisting access window 等显式策略；
- GPU display/copy engine 等并发活动。

对 benchmark 应明确是否测试 cold cache、warm cache，避免把偶然驻留当成算法保证。

## 8. Constant memory

constant memory 是 device 范围的只读地址空间，容量与 cache 行为由架构/规范定义。
其关键性能特征是 warp 内地址一致性：

- 所有 lane 读同一个地址：可以广播，通常很高效；
- warp 内访问少数不同地址：按唯一地址数服务；
- 32 个 lane 读 32 个不同地址：可能显著序列化。

适合小型、只读、warp-uniform 的参数或查表，不适合把任意只读大数组都搬进去。

## 9. Texture/read-only 路径

texture 路径支持特定 cache、寻址、格式转换与空间局部性语义。它在二维空间邻近但
不一定线性连续的访问中可能有效。

普通 global load 在现代架构上也有缓存。选择 texture 的理由应来自访问模式或所需
寻址功能，而不是“texture 一定比 global 快”。

## 10. Host/device 数据移动

### 10.1 Pageable 与 pinned host memory

- pageable 内存可能需要 runtime staging；
- pinned/page-locked 内存更适合 DMA 和异步传输；
- pinned 内存会占用有限系统资源，过量会影响系统；
- `cudaMemcpyAsync` 想与 kernel 真正 overlap，通常需要 pinned host buffer、
  合适的 stream、独立 copy engine 和无隐式依赖。

### 10.2 Mapped/zero-copy

GPU 可以直接访问映射的 host memory，省去显式 device allocation/copy，但访问经
PCIe/NVLink 等互连，latency 和带宽通常不如本地 device memory。适合少量、一次性、
难以提前搬运的数据，而非反复访问的热循环。

### 10.3 Unified Memory

Unified Memory 提供统一地址与自动迁移，不代表数据同时免费存在于 CPU/GPU：

- 首次访问可能 page fault；
- CPU/GPU 交替写可能来回迁移；
- oversubscription 可能触发 eviction；
- 可用 prefetch 和 memory advice 表达预期。

使用 Nsight Systems 观察 migration/page fault，不要只看 kernel duration。

## 11. 典型数据移动优化层次

以 GEMM/Attention tile 为例：

```text
1. lane 连续访问，先保证 global coalescing
2. block 合作搬运，减少重复 global load
3. shared layout 适配计算，消除 bank conflict
4. register fragment 提高线程内复用
5. double/multi-stage buffer 隐藏 global-to-shared latency
6. cache/block swizzle 改善跨 block 的 L2 reuse
7. 融合 epilogue，减少中间 global round trip
```

每一步都有资源成本，必须用 duration 和计数器验证。

## 12. 面试回答框架

### 请介绍 GPU 内存层次

推荐按“可见性—管理方式—典型用途—风险”回答：

1. register：thread 私有，最快，受 pressure 和 occupancy 约束；
2. local：thread 私有地址空间，但物理在显存路径，常见于 spill；
3. shared：block 可编程片上缓存/通信，注意 bank conflict 和 barrier；
4. L1/L2：硬件管理，L1 通常靠近 SM，L2 全 GPU 共享；
5. global：大容量 device memory，需合并访问和复用；
6. constant/texture：针对特定只读访问模式；
7. host/pinned/Unified Memory：涉及系统互连和迁移。

### 什么是 coalescing？

不是“线程访问连续地址”这句口号，而是一个 warp 的 active lane 地址被硬件组合后，
覆盖尽量少的对齐 sector，使 physical traffic 接近 useful bytes。

### Shared memory 为什么快，为什么也可能慢？

它是片上、软件可控、可复用；但 bank conflict 会把一次 warp 请求拆成多个
wavefront，barrier 和容量也可能降低并行度。

## 13. 参考资料

- [CUDA Best Practices Guide: Memory Optimizations](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#memory-optimizations)
- [CUDA Programming Guide: Writing SIMT Kernels](https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html)
- [Nsight Compute Profiling Guide: Memory Workload Analysis](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)

