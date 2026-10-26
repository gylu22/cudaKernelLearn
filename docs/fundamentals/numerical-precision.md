# CUDA 数值精度、浮点与验证

高性能 kernel 的“正确”不是逐 bit 等于某个 CPU 循环，而是在明确的数据类型、计算
顺序、累加类型、数学模式和容差契约下得到可接受结果。

## 1. 浮点数由什么组成

二进制浮点通常包含：

```text
sign × significand × 2^exponent
```

指数位主要决定范围，fraction/significand 位主要决定相对精度。

常见格式的概念比较：

| 类型 | 主要特点 | 常见用途 |
| --- | --- | --- |
| FP64 | 高精度和范围、吞吐依 GPU 定位差异很大 | 科学计算、reference |
| FP32 | 通用单精度 | 累加、数值敏感路径 |
| TF32 | FP32 范围、较低输入有效精度的 Tensor Core 模式 | FP32 GEMM 加速 |
| FP16 | 范围较小、精度高于 BF16 fraction | 训练/推理存储和 Tensor Core |
| BF16 | FP32 类似指数范围、fraction 较短 | 训练，减少 overflow 风险 |
| FP8/更低精度 | 格式与 scale 策略相关 | 新架构低精度训练/推理 |

具体支持的指令、tile 和吞吐依 compute capability。

## 2. 舍入与 ULP

大多数实数无法被有限二进制位精确表示，每次运算可能舍入。默认常见模式是
round-to-nearest, ties-to-even。

ULP 表示相邻可表示数之间的间隔尺度。数值绝对值越大，FP32 相邻表示的绝对间距
通常越大。因此只用固定 absolute tolerance 不适合覆盖所有量级。

常见比较：

```text
|actual - expected| <= atol + rtol × |expected|
```

还应记录 max absolute、max relative、mean error 和异常值位置。

## 3. 浮点不满足结合律

```text
(a + b) + c != a + (b + c)
```

并行 reduction 改变加法树，因此 GPU 结果与串行 CPU 或不同 block 配置略有差异是
正常现象。atomic reduction 的到达顺序还可能导致多次运行不完全确定。

测试应区分：

- 算法错误；
- 合理舍入差异；
- 并行顺序差异；
- 低精度输入/累加造成的误差；
- fast-math 近似。

## 4. FMA

Fused multiply-add：

```text
fma(a, b, c) = round(a × b + c)
```

乘法和加法之间不做单独舍入，通常比拆开的两条操作更准确，也适合硬件吞吐。它也
意味着 GPU 结果可能不同于禁止 contraction 的 reference。

## 5. Accumulation 类型

输入 dtype 与 accumulator dtype 可以不同：

```text
FP16/BF16 input + FP32 accumulate
TF32 multiply semantics + FP32 accumulate
INT8 input + INT32 accumulate
```

低精度输入即使使用 FP32 累加，也无法恢复输入量化/舍入丢失的信息；FP32 累加主要
降低长 reduction 中继续累积的误差与 overflow 风险。

## 6. Reduction 的稳定性

常见改善：

- pairwise/tree reduction，通常好于严格从左到右累加；
- FP32/FP64 accumulator；
- Kahan/补偿求和，成本更高；
- 先缩放再求和；
- 避免大数与极小数直接混合；
- 根据数据范围选择算法。

高性能 kernel 需要在数值稳定性和吞吐之间明确取舍。

## 7. Softmax 稳定性

直接计算：

```text
exp(x_i) / sum_j exp(x_j)
```

可能 overflow。稳定形式：

```text
m = max_j x_j
p_i = exp(x_i - m) / sum_j exp(x_j - m)
```

Online softmax 跨 tile 维护 running max `m` 与 normalizer `l`：

```text
m_new = max(m_old, tile_max)
l_new = exp(m_old - m_new) × l_old
      + sum(exp(tile_score - m_new))
```

输出 accumulator 也必须按 `exp(m_old-m_new)` 重标定。这是 FlashAttention 正确性
的核心不变量。

## 8. `--use_fast_math`

该选项可能改变：

- 除法、sqrt、reciprocal；
- 部分数学函数使用更快近似；
- contraction/精度相关行为；
- denormal 处理等编译选择。

它不是“让所有 FP32 加法乘法都自动更快”。必须：

1. 独立 backend/构建；
2. 对目标输入范围测误差；
3. 看 SASS/指令是否改变；
4. 看目标 pipeline 与 duration；
5. 不把近似路径和严格路径混在同一性能结论。

## 9. NaN、Inf、subnormal

应测试：

- `+Inf/-Inf`；
- quiet NaN；
- 极小 subnormal；
- 正负零；
- overflow/underflow；
- softmax 全 mask 或极端 score。

不同数学模式和 dtype 对 subnormal 的处理可能不同。生产接口要定义异常输入语义，
而不是只在随机 `N(0,1)` 上验证。

## 10. 确定性

确定性受到：

- atomic 顺序；
- block 调度；
- reduction tree；
- library algorithm 选择；
- workspace；
- 并发 stream；
- compiler/architecture；
- stochastic rounding（若启用）

影响。

“数值在容差内正确”和“bitwise deterministic”是两个不同契约。后者通常限制算法
选择或降低性能。

## 11. 验证策略

### 分层 reference

1. 高精度 CPU/FP64 小 shape；
2. PyTorch/cuBLAS 同 dtype reference；
3. 每个中间阶段的独立 reference；
4. 端到端误差。

### 输入集合

- 随机正常分布；
- 均匀分布；
- 全零/全一；
- 大小量级混合；
- cancellation；
- NaN/Inf；
- tile 边界；
- 很长 reduction。

### 报告

```text
dtype/math mode:
reference:
shape:
atol/rtol:
max_abs:
max_rel:
mean_abs:
NaN/Inf policy:
deterministic requirement:
```

## 12. 面试问题

### FP16 与 BF16 如何选择？

FP16 fraction 更多，但 exponent 范围较小；BF16 范围接近 FP32、fraction 更少。
训练中 BF16 往往减少 overflow/underflow 风险，具体仍取决于硬件、模型和 scale。

### TF32 是什么？

它主要是 Tensor Core 的计算模式，保留 FP32 类似范围但输入有效 fraction 精度较低，
通常 FP32 累加；不是普通 tensor 的通用存储 dtype。

### 为什么 GPU 与 CPU 结果不同不一定是 bug？

并行 reduction 顺序、FMA contraction、数学函数实现和 dtype 路径可能不同。应根据
数值契约和误差分析判断，而非要求逐 bit 一致。

## 13. 参考资料

- [CUDA Programming Guide: Floating-Point Computation](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/mathematical-functions.html)
- [Floating Point and IEEE 754](https://docs.nvidia.com/cuda/pdf/Floating_Point_on_NVIDIA_GPU.pdf)
- [CUDA Best Practices Guide: Numerical Accuracy](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

