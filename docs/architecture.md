# FlashAttention 架构与实现路线

本文只描述 FlashAttention 路线。仓库总览见根目录 README，SGEMM 见
[SGEMM 学习路线](sgemm/index.md)。

当前实现的详细性能分析见
[FlashAttention 优化思路](flash-attention/optimization-guide.md)，分阶段开发计划见
[FlashAttention 后续优化路线](flash-attention/roadmap.md)。

## 设计目标

每个优化阶段都是可独立选择的 backend。只有在满足相同输入契约并通过同一组
correctness 测试后，实验实现才进入默认构建。这样 NCU 对比主要反映算法和数据
布局差异，而不是 shape、stream 或边界行为不同。

## 分层边界

```text
csrc/
├── bindings/                 PyTorch 输入检查、设备保护和 dispatch
├── include/flash_attn/       常量与 backend-neutral launcher 声明
├── kernels/
│   ├── simt/                 FP32 CUDA Core 路线
│   └── tensor_core/          TF32 WMMA 路线
└── legacy/                   不参与默认构建的教学代码和实验
```

- binding 负责 tensor 契约、`CUDAGuard`、current CUDA stream 和错误转换；
- launcher 只接收裸指针、尺寸和 `cudaStream_t`；
- kernel 不依赖 PyTorch；
- kernel-specific tile/layout 不应泄漏到公共 Python API。

## 当前输入契约

- Q/K/V 为同 device、同 shape 的 contiguous FP32 CUDA tensor；
- layout 为 `[batch, heads, seq_len, head_dim]`（BHND）；
- `head_dim=64`，Q/K/V 序列长度相同；
- non-causal forward，无 mask、dropout、GQA/MQA 和 backward；
- 支持非 64 整数倍的 sequence tail；
- launch 到调用者的 PyTorch current stream。

## 活动 backend

| Backend | 执行路径 | Score 存储 | 学习重点 |
|---|---|---|---|
| `simt_fp32_pairwise_reg_scores` | FP32 SIMT | 每线程寄存器 | register tile、shuffle、合并访问 |
| `simt_fp32_pairwise_smem_scores` | FP32 SIMT | shared memory | register/SMEM/occupancy 权衡 |
| `simt_fp32_pairwise_compact_smem_scores` | FP32 SIMT | 紧凑 shared memory | 动态 SMEM 与驻留 Block 数 |
| `tensor_core_tf32_wmma` | TF32 WMMA + FP32 softmax | shared memory/fragment | QK、PV 的 Tensor Core 教学基线 |

前三个 SIMT backend 使用相同的 FP32 数学、64×64 Q/KV tile、D=64 和 pairwise
线程映射，因此适合比较 score 存储策略。WMMA backend 改变了数值路径，必须使用
单独容差，不能把误差变化归因于存储策略。

## Online softmax 不变量

对已经处理的 key 集合维护行最大值 `m`、归一化因子 `l` 和未归一化输出 `o`。
新 tile 的最大值为 `m_t` 时：

```text
m' = max(m, m_t)
l' = exp(m-m') * l + sum_j exp(s_j-m')
o' = exp(m-m') * o + sum_j exp(s_j-m') * v_j
```

最终输出为 `o/l`。实现必须对旧状态做重标定；仅对每个 tile 单独 softmax 再相加
并不等价于全序列 softmax。

## 命名规则

活动 CUDA 文件：

```text
<op>_<dtype>_<sequence>_<thread-map>_<strategy>.cu
```

Python backend：

```text
<family>_<dtype>_<thread-map>_<strategy>
```

`lNN` 表示保留的 learning stage，`xNN` 表示尚未完成的 experiment。新优化应
增加 backend，而不是覆盖旧文件。

## 后续路线

完整计划见 [FlashAttention 后续优化路线](flash-attention/roadmap.md)。近期先处理：

1. 减小每线程 Q/O fragment，并用 sub-warp primitive 完成行归约；
2. 对照 score 存 shared memory 与 score-recompute；
3. 先实现同步两级 K/V pipeline，再引入 `cp.async`；
4. 在 SIMT 数据通路稳定后增加 FP16/BF16 Tensor Core 路径。

## 参考资料

- [FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [PyTorch scaled dot product attention](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
