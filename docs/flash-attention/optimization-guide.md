# FlashAttention 优化思路：从算法到当前实现

本文用于重新建立 FlashAttention kernel 的优化思维。重点不是背某个 tile 参数，而是
知道每一步减少了什么流量、增加了什么资源，以及应当用什么证据判断优化是否成立。
当前公共接口和 backend 见[架构说明](../architecture.md)。

## 1. 先明确目标

普通 attention forward：

```text
S = QK^T / sqrt(D)
P = softmax(S)
O = PV
```

如果把完整 `S` 和 `P` 写入显存，空间和额外显存流量随序列长度按 `N²` 增长。
FlashAttention 的核心不是“近似 softmax”，而是：

1. 将 Q、K、V 分块；
2. 只在片上保存当前 tile；
3. 用 online softmax 合并不同 key tile；
4. 不把完整 `N×N` score/probability materialize 到 HBM。

对已经处理的 key 维护行状态：

```text
m' = max(m, tile_max)
l' = exp(m-m') × l + sum(exp(score-m'))
o' = exp(m-m') × o + sum(exp(score-m') × V)
```

因此优化必须同时保持三个条件：

- `m/l/o` 更新在数学上正确；
- Q/K/V 和中间状态的流量减少；
- tile、线程和同步不会消耗掉节省的成本。

## 2. 当前实现的线程与 tile

### SIMT 路径

三个 SIMT backend 共同使用：

| 项目 | 当前值 |
| --- | ---: |
| Query tile | 64 行 |
| Key/Value tile | 64 行 |
| Head dimension | 64 |
| Threads/block | 128 |
| Threads/query | 2 |
| 每线程处理的 key | 32 |

两个相邻线程共同处理一个 query row。每个线程保留完整的：

- `q_registers[64]`；
- `output_accumulator[64]`；
- softmax 状态和地址临时量；
- register-score backend 还保留 32 个 score。

仅 Q/O 就至少需要 128 个 FP32 标量的逻辑 live range，所以寄存器是当前 SIMT
路线的首要约束之一。已有 `sm_86` 编译记录显示三个 SIMT backend 约使用
238–254 registers/thread；这不能仅靠提高 block 数解决。

### Tensor Core 路径

当前 WMMA backend 使用：

| 项目 | 当前值 |
| --- | ---: |
| Query tile | 16 行 |
| Key/Value tile | 64 行 |
| Threads/block | 128（4 warps） |
| QK/PV | TF32 WMMA |
| Softmax | FP32 标量 |

四个 warp 分别计算 QK 的四个 16-column tile，以及 PV 的四个 16-column 输出
tile。QK/PV 已进入 Tensor Core，但每行 softmax 仍由单个线程遍历 64 个 score，
中间结果需要多次 shared-memory store/load 和 block barrier。

所以“出现 HMMA 指令”只证明 Tensor Core 被使用，不代表整个 attention 已经
Tensor-Core-bound。

## 3. 三个 SIMT backend 在比较什么

### Register scores

`simt_fp32_pairwise_reg_scores` 把当前线程的 score 保留在寄存器。

优点：

- score 不产生 shared-memory 往返；
- 计算和使用距离短。

代价：

- 32 个 score 延长 live range；
- 加上 Q/O accumulator 后寄存器接近架构上限；
- 可能限制 resident blocks/warps。

### Shared-memory scores

`simt_fp32_pairwise_smem_scores` 将 score 写入 shared memory，再在 softmax/PV 阶段
读回。

这是“寄存器换 shared memory”的实验：

- 寄存器可能减少；
- 但多一次 shared store/load 和同步；
- dynamic shared memory 约 51 KiB，可能限制 block residency。

它不保证比 register 版本快，价值在于揭示资源瓶颈发生在哪里。

### Compact shared-memory scores

`simt_fp32_pairwise_compact_smem_scores` 将动态 shared memory 精确压到 49 KiB。
在当前 RTX 3060 Laptop GPU 上，每个 SM 有 102,400 bytes shared memory；考虑
驱动为 block 保留的空间后，49 KiB 设计使“两 block 是否可驻留”成为可能。

但实际 residency 还同时受约 253 registers/thread 限制。shared memory 满足两
block 不代表最终一定有两 block 驻留，应检查 NCU 的 Block Limit、Theoretical
Occupancy 和 Achieved Occupancy。

## 4. 当前最可能的瓶颈

### 4.1 寄存器与长 live range

每线程完整保存 Q/O 是最明显的问题：

```text
64 Q + 64 O + score/state/address
```

表现通常是：

- registers/thread 很高；
- theoretical occupancy 低；
- active warp 不一定能提供足够 latency hiding；
- 强行 `--maxrregcount` 会导致 local-memory spill。

正确方向不是简单限制寄存器，而是改变工作划分：

- 一个 query row 由 4/8/16 个 lane 协作；
- 每 lane 只保存 D 的一个 fragment；
- 用 warp shuffle/reduction 合并 dot、max 和 sum；
- O fragment 分布在多个 lane，最终协作写回。

### 4.2 标量 FP32 QK/PV

SIMT backend 对每个 score 做 64 次标量 FMA，对每个 probability 又更新 64 个输出。
它适合解释算法，但无法匹配 Tensor Core 的矩阵乘吞吐。

优化时应把问题分开：

1. QK 是否已经成为 MMA-friendly tile；
2. softmax 是否并行且数值稳定；
3. probability layout 是否能直接供 PV MMA 使用；
4. layout 转换和 shared-memory 往返是否抵消 MMA 收益。

### 4.3 Softmax 和特殊函数

当前 SIMT 每线程对 32 个 score 调用 `expf`；WMMA 版本由一个线程串行处理一行的
64 个元素。可能出现：

- MUFU/特殊函数吞吐压力；
- softmax 串行依赖；
- Tensor Core 等待 softmax；
- 多次 barrier。

先并行 max/sum reduction，再考虑近似指数或 fast math。改变指数实现会改变数值
路径，必须单独记录误差，不能与线程映射优化混成一次实验。

### 4.4 K/V 数据流水线

每个 key tile 当前采用：

```text
load K/V → barrier → QK → softmax → PV → barrier → next tile
```

没有真正的 global-to-shared 异步 pipeline。后续可以：

1. 先做普通 load 的双 buffer，验证 ring index；
2. 再改成 `cuda::pipeline`/`cp.async`；
3. 在计算 tile `t` 时预取 `t+1`；
4. 用 duration、eligible warp 和 SASS 验证是否真的 overlap。

增加 stage 会增加 shared memory，当前 kernel 本来就受 SMEM/register 约束，所以
pipeline 不能脱离资源占用单独判断。

### 4.5 固定 tile 与 shape

SIMT 的 Q tile 为 64，WMMA 的 Q tile 为 16。shape 会影响：

- grid 是否提供足够 block；
- 最后一个 query/key tile 的无效 lane；
- K/V 被不同 query block 重读的次数；
- 小序列中 launch、barrier 和 setup 的占比；
- 长序列中 online-softmax 循环与内存流水线。

因此不应只用 `N=1024` 得出全局结论。至少测试：

```text
B/H: (1,1), (2,8)
N:   1, 17, 63, 64, 65, 128, 256, 1024, 2048
D:   当前 64；后续加入 32/128
```

## 5. 建议的优化顺序

```text
正确性与 shape 基线
  ↓
减少每线程 Q/O fragment
  ↓
warp/sub-warp reduction
  ↓
决定 score 保存还是重计算
  ↓
并行 softmax，减少 barrier
  ↓
同步 K/V 双缓冲
  ↓
cp.async 多级 pipeline
  ↓
FP16/BF16 Tensor Core QK + PV
  ↓
按 D/N/causal 选择 kernel
```

这个顺序优先消除当前最明显的 register/softmax 限制，再引入异步和低精度。若先在
每线程 128+ 标量 live range 的结构上叠加更多 stage，通常只会让资源压力更难解释。

## 6. 每次实验要回答的问题

| 改动 | 必看证据 |
| --- | --- |
| 减少 fragment | registers、spill、resident blocks、duration |
| score 重计算 | SMEM/寄存器下降是否大于新增 FMA 成本 |
| warp reduction | shuffle 指令、barrier、eligible warps |
| async K/V | SASS copy 指令、memory dependency stall、overlap |
| Tensor Core | HMMA/MMA、tensor pipe、layout/shared 流量 |
| tile 变化 | grid waves、tail、K/V 重读、跨 shape 稳定性 |
| fast math | max/mean error、极端输入、duration |

优化的最终判据仍是相同语义和容差下的 duration，而不是某个 occupancy 或 SOL
百分比单独变好。

## 7. 当前实现距离生产级的主要差距

- 只有 FP32 BHND、`D=64`、Q/K/V 等长；
- 只有 non-causal forward；
- 没有 mask、bias、dropout、GQA/MQA、ragged sequence；
- 没有 backward、LSE 输出或 autograd 注册；
- 没有按 GPU、dtype、D、N、causal 自动 dispatch；
- WMMA 只覆盖 Ampere+ 的 TF32 教学路径；
- 缺少 graph capture、长时间压力、compute-sanitizer 和多 GPU 测试；
- 没有针对数值精度、确定性和不同数学模式的正式契约。

详细落地顺序见[FlashAttention 后续优化路线](roadmap.md)。

## 参考资料

- [FlashAttention paper](https://arxiv.org/abs/2205.14135)
- [FlashAttention-2 paper](https://arxiv.org/abs/2307.08691)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA asynchronous data copies](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#asynchronous-data-copies)
- [PyTorch scaled dot product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)

