# SGEMM：从正确实现到性能实验

SGEMM 计算行主序 FP32 矩阵乘法：

```text
A[M, K] × B[K, N] -> C[M, N]
C[m, n] = sum_k A[m, k] * B[k, n]
```

它是学习 coalescing、shared-memory tiling、register blocking 和流水线的基础，也直接帮助理解 FlashAttention 中的 QK 与 PV 子问题。

建议先用本文了解接口和运行方式，再阅读：

- [SGEMM 优化思路与性能差距分析](optimization-guide.md)：解释当前各阶段的作用，
  以及不同 shape 与 cuBLAS 差距不同的根因；
- [SGEMM 生产化差距与后续路线](production-roadmap.md)：整理一般 shape、API、
  kernel dispatch、可靠性和性能回归等工作。

## 1. 当前接口契约

```python
from cuda_kernel_lab import sgemm

c = sgemm.forward(a, b, "simt_fp32_03_transposed_smem_vec4")
```

绑定层位于 [`csrc/bindings/sgemm_api.cpp`](../../csrc/bindings/sgemm_api.cpp)，当前要求：

- `a`、`b` 是同一 CUDA device 上的二维 FP32 tensor；
- 两者是 contiguous row-major，且 `a.shape[1] == b.shape[0]`；
- M、N、K 为正数且不超过 launcher 的 32 位索引范围；
- 运算使用当前 PyTorch CUDA stream；
- 返回新分配的 `C[M, N]`，不支持转置、batch、`alpha`/`beta` 或 autograd。

## 2. 活动 backend

| Backend | 优化阶段 | shape 约束 |
| --- | --- | --- |
| `simt_fp32_00_naive` | 每线程一个输出元素 | 任意正 M/N/K |
| `simt_fp32_01_smem_tiled` | 32×32 shared-memory tile | 任意正 M/N/K |
| `simt_fp32_02_thread_tile_vec4` | 8×8 register tile、`float4` 搬运 | M/N 为 128 倍数，K 为 8 倍数 |
| `simt_fp32_03_transposed_smem_vec4` | 转置 A 的 shared-memory 布局 | 同上 |
| `simt_fp32_04_double_buffer` | 普通 load 的逻辑双缓冲 | 同上 |
| `simt_fp32_05_padded_smem` | 带 padding 的教学变体 | 同上 |

后四个 backend 的 128×128×8 约束由绑定层显式检查，避免旧 kernel 越界。历史函数名目前仍是 dispatch alias，但新代码和实验记录应使用规范 backend 名。

`padded_smem` 的名称只表示布局中存在 padding，不代表所有 bank conflict 已消除；是否有收益必须按具体访问模式用 profiler 验证。

## 3. 构建与验证

从仓库根目录安装：

```bash
python -m pip install -v -e .
```

运行接口和 CUDA 正确性测试：

```bash
python -m pytest -q tests/test_python_registry.py
python -m pytest -q tests/test_sgemm_correctness.py
```

测试会覆盖通用 shape、tile 对齐 shape、非默认 stream 和错误输入。FP32 不同累加顺序可能产生舍入差异，应使用明确的 `atol`/`rtol`，不能只比较少量输出或只看平均误差。

## 4. Benchmark

```bash
python benchmarks/benchmark_sgemm.py \
  --m 4096 --n 4096 --k 4096 \
  --warmup 20 --iterations 100 --repeats 5 \
  --output-json results/sgemm/4096-cube.json
```

基准中关闭 PyTorch matmul 的 TF32，避免将手写严格 FP32 SIMT 与 Tensor Core TF32 混为同一精度路径。比较结果时还应说明：

- GPU、driver、CUDA、PyTorch 和时钟状态；
- shape、backend、warmup 与迭代次数；
- 是否包含输出分配和零初始化；
- cuBLAS/PyTorch reference 的精度设置；
- 使用 median、P90/P99 还是平均值。

理论 FP32 工作量为 `2 × M × N × K` FLOP：

```text
TFLOP/s = 2 × M × N × K / elapsed_seconds / 1e12
```

## 5. Profiling

先按 [Profiling 工作流](../profiling/01-workflow.md)定位目标，再采集单个 backend：

```bash
mkdir -p artifacts/ncu

ncu --set full \
  --kernel-name regex:sgemm \
  --launch-skip 10 --launch-count 1 \
  -o artifacts/ncu/sgemm-simt-fp32-03 \
  python benchmarks/benchmark_sgemm.py \
    --m 4096 --n 4096 --k 4096 \
    --implementation simt_fp32_03_transposed_smem_vec4
```

每一步只改变一个因素，并回答：

1. duration 是否真的下降；
2. global load/store 是否合并；
3. shared-memory bank conflict 是否减少；
4. register/shared-memory 增长是否限制 resident warps；
5. eligible/issued warps 与主要 pipeline 是否改善；
6. 结论是否在多组 shape 上成立。

可用[实验记录模板](../experiment-template.md)在 `results/` 保存命令和证据，
原始 `.ncu-rep` 留在被忽略的 `artifacts/`。

## 6. 后续路线

完整规划见 [SGEMM 生产化差距与后续路线](production-roadmap.md)。近期优先级是：

1. 去掉 `beta=0` 路径不必要的输出读取和零初始化，建立公平 benchmark；
2. 建立方阵、瘦长、小 K、大 K 和边界维度的 shape sweep；
3. 为向量化 backend 添加 tail-safe 版本，与对齐快路径分开；
4. 增加多组 tile 和 shape dispatcher；
5. 在干净基线上验证 `cp.async` 多级流水。

旧实现的设计与修正后的 bank 分析见[历史 kernel 笔记](legacy-kernel-notes.md)。
