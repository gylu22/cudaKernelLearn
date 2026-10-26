# 7. 从 CUDA Source 到 PTX/SASS 的指令级归因

profiling 的最后一公里是回答：哪条源码产生了当前流量或等待？编译器实际生成了什么
指令？修改源码后，目标机器指令是否真的变化？

## 1. 三个层级

```text
CUDA C++
  ↓ nvcc front-end / optimization
PTX：虚拟 ISA
  ↓ ptxas 或 driver JIT
SASS：目标 GPU 机器指令
```

PTX 不是最终执行指令。验证 vector load、Tensor Core、async copy、spill 或 fast-math
路径时，以目标架构 SASS 为最终证据。

## 2. 构建信息

推荐 release profiling 构建保留：

```text
-O3
-lineinfo
--ptxas-options=-v
明确目标 SM
```

区别：

- `-lineinfo`：保留源码行映射，通常适合性能分析；
- `-G`：device debug，会显著改变优化与性能，不用于性能结论；
- `-Xptxas -v`：输出 register、spill、stack 等；
- `--keep`/相关选项：需要时保留中间文件。

## 3. 静态工具

列出 cubin/fatbin 信息：

```bash
cuobjdump --list-elf extension.so
cuobjdump --dump-resource-usage extension.so
```

反汇编：

```bash
cuobjdump --dump-sass extension.so
nvdisasm kernel.cubin
```

不同 CUDA 版本输出格式可能变化。大型 extension 应按 kernel 名过滤。

## 4. NCU Source 页面

Source 页面可以关联：

- CUDA source；
- PTX；
- SASS；
- 指令执行计数；
- branch/predication；
- sampled stall reason；
- memory access 指标。

先确认：

- 当前 kernel 的 cubin 有 line info；
- 展示的是目标 architecture code；
- source 文件路径可访问；
- 指标是 aggregate 还是 instance；
- stall sample 归因到等待的消费者指令。

## 5. 消费者与生产者

若一条 FMA 显示 long scoreboard，真正原因可能是它依赖的 earlier global load：

```text
LDG R20, [address]    // producer
... independent work
FFMA R8, R20, R12, R8 // consumer waits if data not ready
```

stall sample 常落在不能前进的 consumer。分析时沿寄存器依赖向前找 producer，而
不是机械修改高亮行。

## 6. 检查哪些指令

### Global/local load-store

查看：

- 标量还是 64/128-bit；
- 地址是否有额外整数指令；
- local `LDL/STL`；
- cache operator；
- predicate/tail；
- store 是否合并。

### Shared memory

查看：

- shared load/store 宽度；
- 一条源码是否拆成多条指令；
- layout 与 address arithmetic；
- async copy 是否真的生成目标指令；
- barrier/wait 序列。

### 计算

查看：

- FMA 是否生成；
- FP32/FP64/TF32/FP16 路径；
- MMA/WGMMA 等是否与目标架构一致；
- conversion 指令；
- exp/reciprocal/sqrt 的实现；
- 冗余 move/shuffle。

### 控制

查看：

- branch 与 predication；
- loop 是否展开；
- active mask；
- barrier；
- early exit。

## 7. 指令数如何解释

动态 executed instruction 比静态反汇编更重要：

- loop body 静态只有几条，运行数可能很大；
- predicated-off 指令可能被发射但 lane 不做有效工作；
- 一条 warp instruction 不等于 32 个完全有效元素；
- instruction 数下降但 pipeline 更差，duration 仍可能上升。

同时看：

```text
instructions per output
active threads per instruction
target pipeline throughput
issue efficiency
duration
```

## 8. 地址计算

复杂 layout 常引入整数乘加、除法、取模和转换。优化方向：

- 将 invariant 移出循环；
- pointer bump 替代重复完整索引；
- 编译期常量让除法转 shift/multiply；
- 使用合适的 32/64-bit index；
- 避免破坏可读性和范围正确性。

64-bit 地址本身需要足够位宽，但维度索引是否都必须 64-bit 应依据最大 tensor 范围。

## 9. 验证常见源码意图

### `float4`

检查是否出现预期宽度 load/store，以及地址对齐和 tail 分支是否让编译器拆分。

### `#pragma unroll`

检查是否实际展开、register 是否上升、code size 是否膨胀、动态 branch 是否减少。

### `__restrict__`

检查 load 是否能更好调度/复用；同时保证实际不别名。

### WMMA/MMA

确认生成目标 tensor 指令，输入转换和 shared layout 成本也要计入。

### Async copy

确认目标架构生成相应数据移动和 wait/commit；只有指令存在还不代表形成有效 overlap。

## 10. 编译器实验的正确方式

每个实验保存：

```text
source diff
compiler flags/version
register/spill/SMEM
关键 SASS diff
duration
相关 counters
correctness
```

不要只凭 SASS “看起来更短”宣布优化。

## 11. SGEMM 检查点

- A/B global load 是否为预期向量宽度；
- shared A transpose 后地址计算；
- B padding 后 shared load wavefront；
- K-loop 的 barrier 数；
- accumulator 是否保持寄存器；
- epilogue 是否在 `beta=0` 仍加载 C；
- double buffer 是否真的重叠，还是同步搬运。

## 12. FlashAttention 检查点

- Q/O fragment 是否导致 local spill；
- `expf` 对应指令序列；
- row reduction 的 shuffle；
- score 的 register/shared 访问；
- barrier/wait；
- WMMA QK/PV 是否都走 Tensor Core；
- softmax 是否仍由少量线程串行执行。

## 13. 面试问题

### PTX 已经出现 `cp.async`，是否能证明流水优化成功？

只能证明编译路径包含该语义。还需要确认 SASS、stage 依赖、计算/搬运 overlap、
stall、资源使用和最终 duration。

### 为什么 stall 高亮行不一定是根因？

采样通常归因到等待依赖的 consumer，真正造成等待的 producer load 可能在更早位置。

## 14. 参考资料

- [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [Nsight Compute Profiling Guide: Source Counters](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)
- [CUDA Binary Utilities](https://docs.nvidia.com/cuda/cuda-binary-utilities/)
