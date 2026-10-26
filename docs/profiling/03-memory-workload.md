# 3. Memory Workload Analysis

![Memory Workload 表格示例](assets/memory-table.png)

内存分析的目标是把“逻辑访问”对应到“硬件请求与 sector”，再判断这些流量是必要
数据、低效传输，还是 register spill。

## Global memory：请求、sector 与合并访问

现代 NVIDIA GPU 的 L1/TEX 和 L2 统计通常以 32-byte sector 表达。以 warp 中
32 个 lane 各读取一个连续 FP32 为例，有效数据 128 B，理想情况下覆盖 4 个
32-byte sector。实际期望值仍取决于访问宽度、活跃 lane、对齐、cache policy 和
架构，不能把“4 sectors/request”套给所有指令。

排查顺序：

1. 分开看 load 与 store；
2. 比较 requested bytes 与 transferred sectors；
3. 检查地址是否由 `threadIdx.x` 映射到连续元素；
4. 检查首地址和向量访问的自然对齐；
5. 回到 Source/SASS 找产生高 sector/request 的指令。

向量类型不会自动修复跨线程的 stride；`float4` 还要求地址满足相应对齐，并要处理
tail。

## Shared memory：bank conflict

在常见现代架构中 shared memory 有 32 个 bank，连续 32-bit word 映射到连续
bank。一个 warp 指令中：

- lane 访问不同 bank，可并行服务；
- 多 lane 读取同一地址可广播，不属于 bank conflict；
- 多 lane 访问同一 bank 的不同地址会被拆分为多个 wavefront。

![Bank conflict 示例](assets/bank-conflict.png)

padding 或转置只有在改变了实际 bank 映射时才有效。使用
`address_in_32bit_words % 32` 推导后，再用 NCU 的 shared wavefront/conflict
指标验证；不要因为数组第二维加了常数就假定冲突消失。

## Local memory 与 spill

“局部变量”不一定在 local memory：编译器优先放寄存器。寄存器不足、动态索引的
数组或较大线程私有对象可能进入 local memory；其物理存储在 device memory，并经
过缓存层级。

同时检查：

- `ptxas` 的 spill load/store bytes；
- NCU local load/store 指令和 sector；
- Source 页面是否出现意外的 `LDL`/`STL`；
- 限制寄存器后 duration 是否反而变差。

## Cache hit rate 的边界

低 hit rate 对一次性 streaming 访问可能完全正常；高 hit rate 也可能掩盖大量
重复或无效请求。优先看总流量、sector efficiency 和数据复用是否符合算法预期，
再解释 hit rate。详见[缓存与未命中](cache.md)。

## 从算法字节到物理流量

按四层记账：

```text
算法有效元素
  → warp memory instructions
      → request / sector / wavefront
          → L1 miss / L2 hit / DRAM bytes
```

示例：32 lane 连续加载 FP32，useful bytes 是 128 B。理想对齐时常覆盖 4 个 32 B
sector。若 stride 让每 lane 落到不同 sector，useful bytes 仍是 128 B，但 sector
流量可能大幅上升。

分析时为 load/store 分别计算：

```text
useful_bytes
requested_bytes
L1 sectors
L2 sectors
DRAM bytes
```

不要把某一级 sector 直接当成另一级实际传输，cache hit 会改变下游流量。

## Memory latency 与 bandwidth 的区分

### 更像 bandwidth saturation

- 对应层级 `% peak sustained` 高；
- 有许多请求在途；
- scheduler 仍持续 issue；
- 减少 bytes 后 duration 下降。

### 更像 latency exposure

- 总带宽不高；
- eligible warp 少；
- long/short scoreboard 高；
- dependency chain 紧；
- 增加 ILP/MLP/warp 或预取后改善。

## Store 也需要分析

常被忽略的问题：

- C/output store 是否连续；
- partial result 是否多次写回；
- `beta=0` 是否仍加载旧 C；
- `torch::zeros` 是否额外产生 memset；
- atomic store 是否热点；
- kernel 结束前 drain 是否来自大量 store。

## Shared-memory 分析步骤

1. 写出每个 lane 的 word address；
2. 计算 bank；
3. 确认指令访问宽度；
4. 区分 broadcast 与同 bank 不同地址；
5. 查看 ideal/actual/excessive wavefront；
6. 修改 layout 后检查 SMEM 占用和 occupancy；
7. 最终比较 duration。

## 本仓库检查清单

SGEMM：

- A/B global load 是否按 warp 合并；
- thread tile 是否引入非对齐或 tail 越界；
- shared A/B 读是否广播或冲突；
- C store 是否连续；
- 大寄存器 tile 是否 spill。

FlashAttention：

- Q/K/V tile 是否重复从 DRAM 读取；
- score、概率和输出中间量是否落入 local memory；
- dynamic SMEM 是否降低 block residency；
- tail 分支是否只影响最后少量 tile。

## 参考资料

- [CUDA Best Practices Guide: Memory Optimizations](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
- [Nsight Compute Profiling Guide: Memory Workload Analysis](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)
