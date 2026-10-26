# 统一实验基础设施

本页定义 FlashAttention 与 SGEMM 共用的 benchmark、结果记录和 profiling 契约。
目标不是让所有算子使用同一个 kernel 接口，而是让不同实验产生可比较、可追溯的
证据。

## 1. Backend 元数据只有一个来源

公共 backend 名称、默认实现、tile 约束和数值容差分别定义在：

- `cuda_kernel_lab/flash_attention.py`
- `cuda_kernel_lab/sgemm.py`

benchmark 和 profiling CLI 从这里生成 `choices`，不再维护独立列表。新增 backend
时先注册元数据，再增加 native dispatch 和测试；否则 CLI 与实际扩展容易漂移。

## 2. 统一计时语义

`benchmarks/common.py` 中的 `benchmark_cuda` 执行：

1. 若干次不计时 warmup；
2. 同步一次，隔离初始化工作；
3. 每个 sample 用同一 stream 上的 CUDA events 包围多次调用；
4. 将 sample 总时长除以 iterations；
5. 报告多组 sample 的 median、minimum、maximum 和原始值。

CUDA events 包围多次 native extension 调用；每次调用仍会创建输出，相关的设备端
初始化与 kernel 工作处于 event 区间内，但这不是完整的 CPU wall time。输入生成和
reference 计算不在计时范围。两种算子的 JSON 都会明确记录这一口径。

## 3. Benchmark

FlashAttention：

```bash
python benchmarks/benchmark_forward.py \
  --seq-len 64 256 1024 \
  --warmup 20 --iterations 100 --repeats 5 \
  --output-json results/flash-attention/forward.json
```

SGEMM：

```bash
python benchmarks/benchmark_sgemm.py \
  --m 4096 --n 4096 --k 4096 \
  --warmup 20 --iterations 100 --repeats 5 \
  --output-json results/sgemm/4096-cube.json
```

`--implementation` 可选择一个或多个 backend；省略时只运行该算子的默认 backend，
避免无意中对所有教学实现执行大规模长测。不满足 tile 约束的 SGEMM backend 会写入
`skipped` 记录，而不是静默失败。

## 4. JSON 结果

当前 schema version 为 `1`，顶层包含：

```text
schema_version
benchmark
environment
config
results[]
```

环境信息包括 Python、PyTorch、CUDA runtime、cuDNN、GPU、compute capability、
Git commit 和 dirty state。仓库尚无 commit 时 `git_commit` 为 `null`，这是有效
状态，但正式性能结论应基于可定位的 commit。

每个成功结果包含完整 timing samples、误差；SGEMM 还包含 TFLOP/s。JSON 适合后续
自动比较，解释性结论仍记录在[实验模板](experiment-template.md)生成的 Markdown
摘要中。

## 5. 单 launch profiling

统一脚本支持两条路线：

```bash
./profile.sh flash simt_fp32_pairwise_smem_scores 1024
./profile.sh sgemm simt_fp32_03_transposed_smem_vec4 4096 4096 4096
```

报告写入被 Git 忽略的 `artifacts/ncu/`。`PYTHON_BIN` 可显式选择装有 PyTorch 和
扩展的解释器：

```bash
PYTHON_BIN=/path/to/environment/bin/python ./profile.sh flash
```

旧的 `./profile.sh <flash-backend> <seq-len>` 调用仍兼容。

## 6. 自动质量检查

```bash
python scripts/check_docs.py
bash -n profile.sh
python -m compileall -q cuda_kernel_lab benchmarks tests scripts
python -m pytest -q tests/test_python_registry.py
```

GitHub Actions 在普通 CPU runner 上执行以上检查，不尝试编译 CUDA。完整 CUDA
构建和测试需要带 toolkit/GPU 的 self-hosted runner，并在仓库变量中设置
`ENABLE_CUDA_CI=true` 后启用。

## 7. 新 backend 的完成标准

1. registry、native dispatch、文档使用同一个规范名称；
2. binding 检查 dtype/device/layout/shape，并使用 current stream；
3. correctness 覆盖正常 shape、tail、错误输入和非默认 stream；
4. benchmark 输出 JSON 并通过 reference 容差；
5. 单 launch profiling 可以按 backend 精确筛选；
6. 实验摘要记录 commit、环境、命令和 profiler 文件名。
