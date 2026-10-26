# FlashAttention 后续优化路线

本路线针对当前 Ampere `sm_86` 学习环境排序。每个里程碑都应新增 backend，不覆盖
已有实现，并通过相同 correctness、benchmark 和 profiling 流程。

## M0：冻结基线

目标：以后能回答“优化来自哪里”。

- 固定当前四个 backend 的寄存器、SMEM、误差和延迟；
- 保存 `(B,H,N,D)` 测试矩阵，而不是单个 `N=1024`；
- 记录 PyTorch SDPA 后端、TF32/fast-math 和 GPU 时钟状态；
- 为每个结果保存统一 JSON 与实验摘要。

完成标准：

- 所有测试通过；
- 至少覆盖 tile 边界 `63/64/65`；
- 每个结论可定位到 Git commit 和 profiler 报告。

## M1：降低每线程 fragment

目标：解决当前 238–254 registers/thread 的结构性来源。

建议新增：

```text
simt_fp32_subwarp_fragment
```

设计方向：

- 4 或 8 个 lane 共同处理一个 query；
- Q/O 按 D 维分片到 lane；
- 用 shuffle 完成 dot/max/sum；
- 保持 FP32 和现有 online-softmax 语义。

完成标准：

- 无 spill；
- registers/thread 显著下降；
- 多数代表 shape 不慢于当前最佳 SIMT；
- 非默认 stream 和 tail 测试通过。

## M2：Score 策略实验

分别实现：

```text
simt_fp32_subwarp_smem_scores
simt_fp32_subwarp_recompute_scores
```

重计算版本在 softmax/PV 阶段重新执行部分 QK，用计算换 score 存储。需要比较：

- score SMEM 和 barrier 是否减少；
- 新增 FMA 是否被更高 occupancy/更少 SMEM 抵消；
- 短 N 与长 N 的分界点。

完成标准不是确定一个永远最快的方案，而是得到按 shape 可解释的选择规则。

## M3：K/V 双缓冲和异步拷贝

分两步：

1. 普通 load + 双 shared buffer；
2. `cuda::pipeline`/`cp.async` + 2/3 stage。

每一步单独作为 backend。检查：

- stage ownership 和 tail zero-fill；
- shared-memory 增长后的 block residency；
- SASS 中目标 copy 指令；
- memory dependency stall 和 duration。

若寄存器/SMEM 导致 occupancy 明显恶化，应回到 M1/M2，而不是继续堆 stage。

## M4：实用 Tensor Core 路径

当前 TF32 WMMA 是教学基线。更实用的路线是：

```text
FP16/BF16 input
  → FP32 QK accumulator
  → FP32 online softmax
  → FP16/BF16 probability fragment
  → FP32 PV accumulator/output policy
```

重点工作：

- 并行 softmax，替换“一线程处理一行 64 score”；
- 设计 Q/K/P/V shared layout，减少 WMMA 前后转换；
- 将 probability 直接整理为 PV 可消费的 layout；
- 比较 FP16、BF16 的范围和误差；
- 明确输入、累加、输出 dtype。

完成标准：

- QK 和 PV 都有 MMA 指令；
- softmax 不成为明显串行瓶颈；
- 对应 dtype 的容差与极端值测试通过；
- 在代表 shape 上相对 SIMT 有稳定收益。

## M5：功能维度

按以下顺序加入，每项都进入 dispatch key 和测试矩阵：

1. causal；
2. `D=32/128`；
3. Q length 与 KV length 不同；
4. GQA/MQA；
5. additive bias/常见 mask；
6. variable-length/ragged batch；
7. dropout（若面向训练）。

causal 不只是增加一个 `if`：应跳过完全位于因果边界之外的 key tile，并对对角 tile
做 lane predicate，否则会保留大量无效计算。

## M6：Backward

在实现 backward 前决定 forward 契约：

- 是否返回 log-sum-exp（LSE）；
- backward 保存哪些状态、重算哪些状态；
- deterministic 与随机 dropout 状态；
- dQ/dK/dV 的并行归约和原子策略。

完成标准：

- PyTorch autograd 接入；
- gradient reference/`gradcheck` 类测试；
- forward/backward 的显存峰值和端到端训练收益；
- 不只验证单个随机输入。

## M7：生产化与 dispatch

- 建立 `dtype × D × causal × N-range × architecture` dispatch 表；
- 对不支持组合回退到 PyTorch SDPA，而不是报模糊错误；
- 支持 CUDA Graph capture 和多 stream；
- 增加 compute-sanitizer、长时间压力和多 GPU CI；
- 构建多架构 wheel，并明确 driver/toolkit/PyTorch 兼容矩阵；
- 为性能回归设定按 shape 的阈值。

## 优先级建议

在当前仓库中，推荐接下来只做：

1. M0 基线矩阵；
2. M1 sub-warp fragment；
3. M2 score recompute；
4. 再决定进入 M3 还是 M4。

原因是当前最大结构性问题是每线程完整 Q/O fragment。先解决它，后面的 pipeline 和
Tensor Core 布局才有更健康的资源预算。

## 参考资料

- [FlashAttention-2 paper](https://arxiv.org/abs/2307.08691)
- [CUDA WMMA API](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [PTX MMA instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions)
- [libcu++ pipelines](https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives/pipeline.html)
