# SGEMM 历史 kernel 笔记

本页整理合并仓库前的实验代码，保留其学习价值，同时修正原笔记里的线程映射和 bank-conflict 结论。这里的分析针对当前 `csrc/sgemm/kernels/simt/sgemm_fp32.cu`；最终判断仍以实际 SASS 与 Nsight Compute 为准。

## 1. `simt_fp32_00_naive`

每个线程负责一个 `C[row, col]`，沿 K 维顺序累加。

- 优点：结构清楚，可作为正确性 baseline；
- 问题：A 元素跨多个线程重复读取，B 虽可合并访问但没有 tile 复用；
- 实验价值：建立 kernel launch、边界判断和全局内存访问的基准。

## 2. `simt_fp32_01_smem_tiled`

一个 32×32 block（1024 线程）协作加载 A、B 的 32×32 tile。K 方向的最后一个 tile 用边界判断补零，所以支持非 32 倍数的 shape。

对行主序 `float tile[32][32]`，bank 编号可近似写成：

```text
bank = (row × 32 + col) mod 32
```

在协作写入时，warp 内 `threadIdx.y` 固定、`threadIdx.x=0..31`，访问一整行的 32 个 bank。计算时：

- `tileA[ty][k]`：warp 内线程读取同一地址，可由 shared-memory broadcast 服务；
- `tileB[k][tx]`：warp 内线程读取连续 32 个元素，分布到 32 个 bank。

所以这一访问模式本身不需要通过 padding 修复。仍需注意 1024-thread block 可能限制 occupancy，且每线程只计算一个输出，数据复用有限。

## 3. `simt_fp32_02_thread_tile_vec4`

block 为 16×16，即 **256 个线程**，不是旧笔记中的 128 个线程。每线程计算 8×8 输出，整个 block 覆盖 128×128 tile，K tile 为 8。

```text
tid = threadIdx.y × 16 + threadIdx.x,  tid ∈ [0, 255]
```

全局到 shared memory 使用 `float4`，要求对齐并且当前实现只接受 M/N 为 128 倍数、K 为 8 倍数。

### A tile 的读取

原始布局可视为 `tileA[128][8]`，计算访问为：

```text
tileA[ty × 8 + j][i]
address = (ty × 8 + j) × 8 + i
```

一个 warp 包含 `ty=0` 和 `ty=1` 两个半 warp。固定 `i,j` 时，每个半 warp 内是一次 broadcast；两组地址相差 64 个 `float`，映射到同一 bank，因此可能表现为两路冲突。

### B tile 的读取

`tileB[8][128]` 的访问为：

```text
tileB[i][tx × 8 + j]
address = i × 128 + tx × 8 + j
```

固定 `i,j` 时，`tx=0,4,8,12` 等映射到同一 bank 且是不同地址；warp 的两个半区还会重复这些访问。因此该模式会产生 bank conflict，具体计数取决于硬件对 multicast/broadcast 的处理和 profiler 指标定义。

## 4. `simt_fp32_03_transposed_smem_vec4`

A 在写入 shared memory 时转置为 `tileA[8][128]`，计算访问变为：

```text
tileA[i][ty × 8 + j]
```

同一个半 warp 仍读取相同地址，但 `ty=0` 与 `ty=1` 的地址只相差 8 个 `float`，落入不同 bank，从而消除上一版本 A 读取中的两路冲突。

代价是写入转置布局时可能引入新的访问模式，所以不能只根据源码索引宣布“完全无冲突”。应比较 shared load/store wavefront 或 conflict 相关指标，并以 duration 为最终结果。

B 的 `tx × 8` 读取模式并没有因为转置 A 自动改变。

## 5. `simt_fp32_04_double_buffer`

该版本准备两组 register/shared-memory 数据，使“计算当前 K tile”和“准备下一 tile”在程序结构上交错。它使用普通全局 load，而不是 `cp.async`。

这是一种逻辑预取；编译器和硬件能否把 load latency 与计算有效重叠，取决于依赖、寄存器压力、指令调度和 occupancy。shared-memory buffer 翻倍还可能降低 resident blocks。因此应检查：

- duration 和 memory dependency stall；
- register 与 shared-memory 用量；
- eligible/issued warps；
- 生成的 SASS 中 load 与 FMA 的交错。

如果改成 Ampere 的异步 global-to-shared copy，参见[异步拷贝](../tooling/async-copy.md)与[多级流水线](../tooling/multistage-pipeline.md)。

## 6. `simt_fp32_05_padded_smem`

padding 会改变相邻行的 bank 起点，但不能无条件消除冲突。以 B 的固定行访问为例，列仍为 `tx × 8 + j`；只给行末增加一列不会改变同一行内线程之间的 stride 8。

因此该 backend 应理解为“验证 padding 对当前完整布局和编译结果的影响”，而不是已证明的最终优化。至少比较：

- shared load/store conflict 或 wavefront 指标；
- instruction 数与地址计算开销；
- occupancy 限制；
- 最终 kernel duration。

## 7. legacy 目录

`csrc/sgemm/legacy/` 中的 WMMA、swizzle、cuBLAS wrapper 和原 benchmark 不参与默认构建。它们保留作为实验材料，不能视为当前公共 API 或已验证实现。

把旧实验提升为活动 backend 前，必须：

1. 拆分为只接收裸指针、尺寸和 `cudaStream_t` 的 launcher；
2. 通过统一绑定层检查 dtype/device/layout/shape；
3. 使用当前 stream，并补充非默认 stream 测试；
4. 加入多组 shape 的数值测试；
5. 与准确配置的 PyTorch/cuBLAS reference 比较；
6. 为性能结论保存可复现 profiler 记录。

## 8. 分析原则

- bank conflict 是一次 warp shared-memory 指令的地址映射问题，不能只看数组是否 padding；
- broadcast、multicast 和不同地址落入同一 bank 要区分；
- source-level 推导是提出假设，编译后的指令和硬件计数器才是证据；
- 某项计数器改善但 duration 不变，不足以证明优化成功。

