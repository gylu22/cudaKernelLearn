# CUDA 基础与性能模型

本文是本仓库需要反复查阅的概念地图。硬件上限、cache 行为和 Tensor Core 数据类型
会随架构变化；精确数字应查目标 GPU 的 Compute Capability 和 CUDA 文档。

本页适合快速回顾；需要系统学习时按下面顺序阅读：

1. [GPU 硬件与执行模型](gpu-architecture.md)
2. [GPU 内存层次](memory-hierarchy.md)
3. [同步、内存模型、Stream 与并发](execution-and-synchronization.md)
4. [Kernel 性能模型](performance-model.md)
5. [数值精度](numerical-precision.md)
6. [Kernel 开发方法](kernel-development.md)
7. [CUDA 面试复习提纲](interview-review.md)

## 1. 执行模型

### Grid、Block、Thread

kernel 启动创建一个 grid，grid 由 thread block 组成，block 由 thread 组成。

- block 是调度和 shared-memory 协作边界；
- 同一 block 的线程可用 shared memory、`__syncthreads()` 和 cooperative groups
  协作；
- 普通 kernel 中不同 block 的执行顺序不保证，不能依赖跨 block 隐式同步；
- grid/block 可以是一至三维，维度只影响索引表达，不改变内存自动布局。

### Warp 与 SIMT

warp 是调度器管理的一组线程，当前 NVIDIA CUDA 模型中 warp size 为 32。warp
执行同一指令流，但各 lane 有独立寄存器和 predicate。

warp 内条件不一致会产生 divergence：不同路径被 predication 或分批执行，直到
控制流汇合。分支是否昂贵取决于两条路径工作量和编译器生成代码，而不是看到
`if` 就必然慢。

### 调度层次

block 被分配到 SM；warp 在所属 SM sub-partition 上驻留到执行结束。调度器每周期
从 eligible warp 中选择指令发射。区分：

- active/resident：已获得资源；
- eligible：下一条指令依赖满足且管线可用；
- issued：本周期实际发射。

这三个概念是理解 occupancy 与 stall 的基础。

## 2. 内存层次

| 空间 | 可见范围 | 生命周期 | 典型用途 |
|---|---|---|---|
| register | 单线程 | thread | 累加器、地址、临时值 |
| local memory | 单线程地址空间 | thread | spill、动态索引私有数组 |
| shared memory | block/cluster（依 API） | block/cluster | 可编程 tile 与线程通信 |
| global memory | device/进程上下文 | allocation | 输入、输出、长期数据 |
| constant memory | device，只读 | module | warp 内统一地址的只读常量 |
| L1/TEX、L2 | 硬件管理 | 硬件管理 | global/local/texture 请求缓存 |

“local”描述可见性，不表示物理上靠近核心；local memory 通常位于 device memory。

### 合并访问

warp 的 global memory 指令会根据 lane 地址合并成一个或多个内存事务。目标不是
机械使用 `float4`，而是让活跃 lane 覆盖尽量少、对齐的 sector：

```cpp
int i = blockIdx.x * blockDim.x + threadIdx.x;
out[i] = in[i];  // 相邻 lane 访问相邻元素
```

stride、随机索引、未对齐向量和 tail 都会增加事务或需要额外处理。向量化可减少
指令数，但不自动改善跨 lane 的地址分布。

### Shared-memory bank

常见现代架构有 32 个 bank，连续 32-bit word 映射到连续 bank。同一 warp 指令中
访问同一地址可广播；访问同一 bank 的不同地址会序列化为多个 wavefront。

矩阵转置常用 padding：

```cpp
__shared__ float tile[32][33];
```

但 padding 是否有效取决于实际访问公式。应计算 bank index 并用 NCU 验证。

### Constant、texture 与只读数据

constant memory 在 warp lane 读取相同地址时最有效；lane 读取多个不同 constant
地址会被分批服务。Texture/read-only 路径适合特定空间局部性和寻址语义，不应仅因
“只读”就盲目使用。

### Pinned、mapped 与 Unified Memory

- pageable host memory 的异步 H2D/D2H 常需要 runtime staging；
- pinned host memory 支持更可控的异步传输，但锁页是有限系统资源；
- mapped/zero-copy 让 GPU 直接访问 host memory，避免显式拷贝，但延迟和带宽通常
  不如 device memory，适合少量或一次性访问；
- Unified Memory 提供统一虚拟地址和迁移/按需分页；易用不等于迁移免费，应分析
  page fault、prefetch 和访问归属。

## 3. 同步、内存顺序与 Stream

### `__syncthreads()` 与 fence

- `__syncthreads()` 是 block 级 barrier，并提供 block 内相关内存可见性；
- `__syncwarp(mask)` 是参与 lane 的 warp 级同步；
- `__threadfence_block/device/system` 约束调用线程内存操作的可见顺序，不是线程
  rendezvous，也不会让其他线程自动停下来。

所有线程必须以合法的一致控制流到达 block barrier，否则可能死锁或产生未定义行为。

### CUDA Stream

同一 stream 内操作按序，满足条件的不同 stream 操作可以重叠。想获得 H2D/D2H 与
kernel overlap，通常需要 pinned host memory、支持并发的硬件、无隐式同步以及合理
的 stream/依赖事件。

PyTorch extension 必须使用调用者 current stream，而不是硬编码 default stream；
本仓库通过 `c10::cuda::getCurrentCUDAStream(device)` 获取。

### 异步错误

kernel launch 对 host 通常异步：

```cpp
kernel<<<grid, block, 0, stream>>>(...);
CUDA_CHECK(cudaPeekAtLastError());       // launch/configuration error
CUDA_CHECK(cudaStreamSynchronize(stream)); // 调试边界：执行期错误
```

库代码不应为方便检查而在每次 launch 后 `cudaDeviceSynchronize()`，这会破坏异步
语义。生产 binding 可返回 launch error，测试/调试在明确边界同步。

## 4. 性能模型

### TLP、ILP 与 latency hiding

- TLP：更多 eligible warp 供调度器切换；
- ILP：同一线程/warp 中存在更多相互独立的指令；
- MLP：保持多个独立内存请求在途。

提高 thread 数、展开循环或增加 thread tile 都可能增加其中一种并消耗寄存器/
shared memory。优化是在这些资源间取平衡。

### Occupancy

Occupancy 是 active warp 与架构最大 warp 的比值。它限制延迟隐藏能力，却不是性能
目标。低 occupancy 但高管线利用率可以很快；高 occupancy 但所有 warp 等同一类
依赖仍会很慢。详见 [Occupancy](../profiling/occupancy.md)。

### Block size

没有通用最优 block size。候选通常取 warp 的整数倍，再结合：

- 每 block 工作量和边界浪费；
- registers/thread、SMEM/block；
- block/warp/thread 驻留上限；
- grid 是否至少提供多个 wave；
- 实测 duration。

可用 CUDA Occupancy API/NCU calculator 生成候选，但最终仍需 benchmark。

### Roofline

Roofline 用 FLOP/byte 将算法置于计算上限和带宽上限之间。它帮助选择优化方向，
但无法单独解释同步、依赖和负载不均。详见 [Roofline](../profiling/05-roofline.md)。

## 5. SGEMM 优化层次

对 `C[M,N] = A[M,K] × B[K,N]`，常见演进为：

1. 每线程一个 C 元素；
2. global access 合并；
3. block tile 将 A/B 复用到 shared memory；
4. thread tile 在寄存器中累加多个输出；
5. 向量化加载和适配的 shared layout；
6. double/multi-stage pipeline；
7. Tensor Core WMMA/MMA；
8. split-K、persistent tile 或 shape-specific 调度（按 workload 决定）。

SGEMM 约需 `2MNK` FLOP，但性能不能只用这个分子解释；还要记录输入输出字节、
padding/tail、预处理和数值精度。

## 6. 常见并行模式

### Reduction

典型层次是线程局部累加 → warp shuffle 归约 → 每 warp 写 shared → 一个 warp
完成 block 归约。使用 `__shfl_down_sync` 时 mask 必须准确描述参与 lane。

### Scan

高效 scan 通常使用 work-efficient up-sweep/down-sweep 或 warp-level primitive，
再分层处理 block totals。是否需要 exclusive/inclusive、稳定顺序和跨 block 合并
必须在接口中明确。

### Atomic

atomic 保证指定地址更新的原子性，不自动提供整个算法需要的顺序或 barrier。冲突
集中时会串行化；可先做 warp/block 局部聚合，再减少 global atomic 数。支持的类型
和作用域取决于架构与 API。

### `__restrict__`

`__restrict__` 告诉编译器指针在其有效范围内不别名，可能改善 load 缓存和调度。
若实际指针发生别名却作出错误承诺，行为不再可靠；它是语义契约，不是装饰性优化。

## 7. 精度与 Tensor Core

### FP16、BF16、TF32

- FP16：5-bit exponent、10-bit fraction，范围较小；
- BF16：8-bit exponent、7-bit fraction，范围接近 FP32、精度较低；
- TF32：Tensor Core 计算模式，使用 FP32 范围和约 10-bit fraction 精度，通常以
  FP32 accumulator 执行矩阵乘加；它不是通用的内存存储格式。

Tensor Core 支持的数据类型、tile shape、对齐和吞吐依架构变化。不要写死“维度
必须是 8 或 16 的倍数”作为跨架构规则；库可能处理 tail，但最快路径通常偏好特定
对齐和 shape。

### API 层级

从高到低常见为：

1. cuBLAS/cuBLASLt、cuDNN；
2. CUTLASS/CuTe；
3. WMMA C++ API；
4. PTX `mma.sync` / architecture-specific instructions；
5. SASS（通常由工具检查，不建议作为可移植源码接口）。

手写 WMMA 比 cuBLAS 慢很常见：库拥有 shape-specific tactic、layout、pipeline、
split-K、epilogue fusion 和多年调优。教学实现的目标首先是解释数据通路。

### 2:4 结构化稀疏

支持的架构和数据类型上，Tensor Core 可利用满足特定分组约束的结构化稀疏权重。
理论压缩不等于端到端 2× 加速：还受 metadata、非 GEMM 部分、shape 和库路径影响。

## 8. AI Kernel 相关概念

### Kernel fusion

融合可减少 launch 和中间 tensor 的 global-memory round trip，但也会扩大 live
range、增加 register/SMEM、降低 occupancy，并可能减少调度灵活性。是否有益要测量。

### Layout

NHWC/channel-last 有时更适配向量化或 Tensor Core library path，NCHW 在其他算子/
硬件上可能更好。布局选择应基于整张图的转换成本和目标库，而不是通用偏好。

### FlashAttention

FlashAttention 通过 tiling 和 online softmax 避免将完整 `N×N` score/probability
矩阵写回 HBM，减少 IO，同时保持精确 attention 语义（给定相同数值精度模型）。

### KV cache 与 paged layout

自回归推理缓存历史 K/V 以避免重复计算。分页式 KV 管理把逻辑序列映射到非连续
物理块，减少连续大块预留与碎片；代价是地址映射和更复杂的 kernel 访问。

### Triton

Triton 是用于编写并行 GPU program 的语言和编译器。它让开发者表达 program-level
tile，由编译器生成低层代码；它不消除 layout、访存、数值和 profiling 的要求。

### 量化 kernel

线性量化常写为：

```text
q = clamp(round(x / scale) + zero_point)
x_hat = scale * (q - zero_point)
```

关键问题包括 scale 粒度、zero point、累加位宽、饱和/舍入、outlier、layout 与
量化/反量化融合。低比特存储只有映射到目标硬件的高效指令路径才可能带来加速。

## 9. PTX 与可移植性

PTX 是 NVIDIA 的虚拟 ISA，通常由 `nvcc` 生成，再由工具链/JIT 转为目标 SASS。
PTX 比 SASS 更稳定，但具体指令仍受 PTX ISA 版本和 compute capability 约束。
内联 PTX 适合验证编译器未直接表达的能力；应提供架构 guard、正确的地址空间/
constraint，并检查生成 SASS。

## 参考资料

- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA C++ Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [NCCL User Guide](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/)
- [Triton documentation](https://triton-lang.org/main/index.html)
- [FlashAttention paper](https://arxiv.org/abs/2205.14135)
