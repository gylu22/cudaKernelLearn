# 9. SGEMM 与 FlashAttention Profiling 案例推演

本文使用仓库当前实现演示如何从现象形成实验。以下指标方向是待验证假设，不是假装
已经采集过的数值；实际结论必须来自对应 GPU 的报告。

## 案例 1：Naive SGEMM 到 shared-memory tiling

### 静态模型

naive：每个线程计算一个 C，K 循环反复读取 A row 和 B column。相邻线程对 B 的
访问可能合并，但 A 元素会被多个线程重复读取。

shared tiled：A/B tile 合作加载一次，再由 block 内线程复用。

### 假设

- DRAM/L2 输入流量下降；
- shared load/store 增加；
- barrier 增加；
- arithmetic intensity 提高；
- duration 在足够大矩阵下降。

### 采集

```bash
profile naive and smem_tiled with same M/N/K
```

对比：

- correctness；
- duration；
- DRAM/L2 sectors；
- shared requests/wavefront；
- registers/SMEM；
- achieved occupancy；
- SOL。

### 可能反例

小矩阵或小 K 时，barrier 与 shared staging 成本可能大于复用收益。

## 案例 2：Thread tile 为什么提高性能

当前优化 SGEMM 每线程计算 `8×8` 输出。

### 假设

- A/B register fragment 被多个 FMA 复用；
- 每输出的地址/循环/shared load 开销下降；
- ILP 增加；
- registers/thread 明显上升；
- occupancy 可能下降。

### 验证链

```text
register 增加
  但无 spill
  且 eligible warp/issue 足够
  且 FP32 pipeline 提高
  且 duration 下降
→ register blocking 有效
```

不能只用 occupancy 下降宣布优化失败。

## 案例 3：为什么有些 shape 达到 cuBLAS 95%+

### 甜点 shape

- M/N 是 128 的倍数；
- 大型方阵提供很多 block；
- K 足够大摊薄 launch/输出成本；
- `8×8` accumulator 持续复用；
- global vector load 对齐。

### 差距 shape

#### `128×128`

只有一个输出 block。检查：

- grid/waves；
- GPU active；
- launch 与 allocation 占比。

方向不是继续优化单 block FMA，而是专用小矩阵路径或 batching。

#### `4096×128×4096`

只有约 32 个输出 block。检查：

- resident block 与 wave；
- eligible warp；
- 更小 BN；
- split-K。

#### 小 K

检查：

- `torch::zeros`/memset；
- launch；
- K-loop 工作量；
- barrier；
- 大 thread tile 是否过重。

#### 大 K

检查：

- K/BK 循环次数；
- barrier；
- long scoreboard；
- global-to-shared overlap；
- BK/stage sweep。

## 案例 4：Padding 是否真的解决 bank conflict

### 静态推导

写出 shared address：

```text
word_address(lane, inner_index)
bank = word_address % 32
```

分别计算原布局和 padding 后布局。

### 指标

- actual vs ideal shared wavefront；
- excessive wavefront；
- shared throughput；
- short scoreboard/MIO throttle；
- SMEM/block；
- occupancy；
- duration。

### 结论规则

只有 conflict 指标和 duration 都改善，且没有资源副作用，才能称为有效优化。若
conflict 降但 duration 不变，目标 bottleneck 可能不在 shared。

## 案例 5：Double buffer 名称不代表 overlap

当前 SGEMM `double_buffered` 使用两组 buffer，但普通同步 load 不一定与 compute
重叠。

### 检查

- Source/SASS 的 load、barrier、FMA 顺序；
- load 到 first use 的独立指令距离；
- long scoreboard；
- barrier stall；
- eligible warp；
- 与单 buffer duration。

实现 `cp.async` 后还需：

- 目标指令存在；
- wait/commit 正确；
- 下一 stage 在当前 compute 时发起；
- register/SMEM 增量；
- overlap 对 duration 有收益。

## 案例 6：FlashAttention SIMT 的高寄存器

当前实现每线程保留完整 64 元素 Q fragment、64 元素输出 accumulator，部分版本还
保留 score；sm_86 编译记录约 238–254 registers/thread。

### 假设

- register limit 限制 resident block/warp；
- 可能没有 spill，但 latency hiding 受限；
- 缩小每线程 fragment 可以增加并行度；
- 但跨 lane shuffle/shared communication 会增加。

### 实验

做 sub-warp 分工 backend：

- 每 lane 只持有 D 的子片段；
- warp/sub-warp reduction；
- 保持相同数学与 tile。

比较：

- registers/thread；
- spills；
- theoretical/achieved occupancy；
- eligible/issued warp；
- shuffle/shared instructions；
- duration；
- 多 seq_len。

寄存器下降但 shuffle 增多后变慢也是有效实验结论。

## 案例 7：Shared-score 与 register-score

### Register score

- shared traffic 少；
- register pressure 高；
- 可能限制 occupancy。

### Shared score

- register pressure 可能下降；
- shared footprint/traffic 增加；
- bank/wavefront 与 barrier 风险；
- 每 block SMEM 可能限制 residency。

### 正确比较

保持：

- 相同 FP32 数学；
- 相同 tile；
- 相同 thread mapping；
- 相同 tail；
- 相同 benchmark。

然后比较资源、scheduler、memory 与 duration，而不是只看 registers。

## 案例 8：Tensor Core backend 仍可能慢

FlashAttention WMMA 路径的 QK/PV 可以使用 Tensor Core，但 softmax 仍可能由少数
线程串行处理每行多个 score。

检查：

- tensor pipeline throughput；
- WMMA instruction count；
- score staging；
- softmax `expf`/MUFU；
- barrier；
- active/eligible warp；
- QK、softmax、PV 各阶段 source；
- dtype conversion。

Tensor Core 只加速矩阵乘部分；若 softmax、layout 或同步成为新瓶颈，总 kernel 未必
接近 Tensor Core peak。

## 案例 9：一个指标改善但 kernel 变慢

例如 padding 后 excessive wavefront 降低，但：

- SMEM 增加导致 resident block 从 2 降到 1；
- 地址计算增加；
- register 增加；
- 目标本来是 FP32 pipeline bound。

正确结论：

> 该改动确实改善 shared 访问效率，但在这个 shape 上新增资源/指令成本更大，最终
> duration 上升，因此不能作为默认优化。

## 案例 10：完整实验记录示例

```text
Question:
  Why is sgemm_03 slower on M=4096,N=128,K=4096?

Static model:
  grid=32 blocks; fixed BM=BN=128; K iterations=512

Baseline:
  median GPU duration, error, TFLOP/s

Observations:
  low waves; target pipeline not saturated; no spill

Hypothesis:
  output grid is too small; smaller BN or split-K adds useful parallelism

Experiments:
  128x64 tile
  split-K=2/4/8

Validate:
  error, reduction cost, workspace, duration, waves, issue, DRAM bytes

Conclusion:
  state exact winning range and fallback
```

## 11. 参考入口

- [SGEMM 优化思路](../sgemm/optimization-guide.md)
- [FlashAttention 优化思路](../flash-attention/optimization-guide.md)
- [实验记录模板](../experiment-template.md)
