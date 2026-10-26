# CUDA Kernel Lab

这是一个用于学习 CUDA kernel 实现、正确性验证、benchmark 和 profiling 的实验
仓库。目前包含两条相互关联的算子路线：

1. **SGEMM**：从 naive、shared-memory tiling、thread tiling、向量化加载、
   bank-conflict 优化到 double buffer。
2. **FlashAttention Forward**：从 FP32 SIMT、online softmax 逐步走向 WMMA、
   MMA 和 Tensor Core pipeline。

项目不以替代 cuBLAS、PyTorch SDPA 或官方 FlashAttention 为目标。每个优化阶段
保留为独立 backend，复用同一输入契约和测试集，以便归因性能变化。

完整的学习路线、profiling 方法和部署笔记从
[文档中心](docs/index.md)开始阅读。

两条算子的当前优化分析与后续路线：

- [SGEMM 优化思路](docs/sgemm/optimization-guide.md)与
  [生产化路线](docs/sgemm/production-roadmap.md)
- [FlashAttention 优化思路](docs/flash-attention/optimization-guide.md)与
  [后续路线](docs/flash-attention/roadmap.md)

系统学习入口：

- [CUDA 基础知识体系](docs/fundamentals/cuda.md)
- [Kernel Profiling 知识体系](docs/profiling.md)
- [CUDA 面试复习提纲](docs/fundamentals/interview-review.md)

Python 统一入口是 `cuda_kernel_lab`，底层保留两个独立 extension：
`sgemm_cuda` 和 `flash_attn_cuda`。这样可以共享实验设施，同时保持各算子的编译
选项和演进路线互不污染。

## FlashAttention 当前功能范围

- 输入布局：连续的 `[batch, heads, seq_len, head_dim]`
- Q/K/V：形状相同、位于同一 CUDA device 的 FP32 tensor
- `head_dim = 64`
- 支持非 tile 整数倍的 sequence tail
- 使用 PyTorch current CUDA stream
- 当前只支持 non-causal forward
- 暂不支持 mask、dropout、backward、GQA/MQA 和不同 Q/K 序列长度

## 仓库结构

```text
csrc/
├── bindings/
│   ├── flash_attn_api.cpp          FlashAttention 校验和 dispatch
│   └── sgemm_api.cpp               SGEMM 校验、能力约束和 dispatch
├── include/{flash_attn,sgemm}/     各算子的 launcher 契约
├── kernels/{simt,tensor_core}/     FlashAttention 活动实现
├── sgemm/kernels/simt/             SGEMM 活动实现
└── {legacy,sgemm/legacy}/          不参与默认构建的历史实验

cuda_kernel_lab/                    Python API 和 backend 元数据
tests/                              两个算子的 correctness/validation/stream 测试
benchmarks/                         独立 benchmark 和单次 profiling 入口
scripts/                            无 GPU 的仓库与文档质量检查
.github/workflows/                  CPU 门禁和可选 self-hosted CUDA CI
docs/
├── flash-attention/                FlashAttention 优化分析与后续路线
├── fundamentals/                  CUDA 执行与性能基础
├── profiling/                     Nsight Compute 分析路线和截图
├── tooling/                       CMake、PyTorch Extension、异步流水线
├── frameworks/                    ONNX Runtime 与 TensorRT
├── examples/                      可独立练习的教学示例
├── sgemm/                         SGEMM 路线、约束和历史实现分析
└── archive/                       被校订版替代的原始学习笔记
results/                            可版本控制的实验摘要
artifacts/                          本地 profiler 原始报告（Git 忽略）
profiles/ncu/                       已有 Nsight Compute 报告
```

详细的 SGEMM backend、尺寸约束和运行方法见
[SGEMM 文档](docs/sgemm/index.md)。Legacy 文件用于保存学习过程，不直接加入
`setup.py`。

## Kernel 命名规范

### 活动 CUDA 文件

```text
<op>_<dtype>_<sequence>_<thread_map>_<strategy>.cu
```

例如：

```text
fwd_fp32_01_pairwise_smem_scores.cu
```

各字段含义：

| 字段 | 示例 | 含义 |
|---|---|---|
| `op` | `fwd` | forward 操作 |
| `dtype` | `fp32` | 主要输入/计算精度 |
| `sequence` | `01` | 当前实现路线中的顺序编号 |
| `thread_map` | `pairwise` | 两个相邻线程协作处理一个 query row |
| `strategy` | `smem_scores` | score 保存在 shared memory |

目录已经表达了 `simt` 或 `tensor_core` 执行族，因此文件名不重复该字段。

### Python backend

Python dispatch 名称采用：

```text
<family>_<dtype>_<thread_map>_<strategy>
```

例如：

```text
simt_fp32_pairwise_smem_scores
```

Python backend 包含 `family`，因为 SIMT 和 Tensor Core 实现会共享同一个
dispatch namespace。

### Legacy 和实验文件

- `lNN`：legacy learning stage，例如 `l02`
- `xNN`：未完成或探索性 experiment，例如 `x00`

Legacy 编号与活动 kernel 的 `00/01/...` 分离，避免历史教学顺序被误认为当前
性能优化顺序。

## 当前活动实现

| CUDA 文件 | Python backend | Score 存储 | 学习重点 |
|---|---|---|---|
| `fwd_fp32_00_pairwise_reg_scores.cu` | `simt_fp32_pairwise_reg_scores` | 每线程寄存器 | 寄存器 tiling、相邻 lane shuffle、合并访存 |
| `fwd_fp32_01_pairwise_smem_scores.cu` | `simt_fp32_pairwise_smem_scores` | Shared memory | 寄存器压力、SMEM 流量和 occupancy 权衡 |
| `fwd_fp32_02_pairwise_compact_smem_scores.cu` | `simt_fp32_pairwise_compact_smem_scores` | 精确大小的 shared memory | Dynamic SMEM 压到 49 KiB，为驱动保留的 1 KiB/block 留出空间 |
| `fwd_tf32_00_wmma.cu` | `tensor_core_tf32_wmma` | WMMA + FP32 softmax | 使用 TF32 Tensor Core 完成 QK 和 PV |

三个 SIMT 实现使用相同的 FP32 数学、`64×64` Q/KV tile、D=64 和 pairwise 线程映射，
因此 profiling 结果具有可比性。

当前 `sm_86` 编译基线：

| Backend | Registers/thread | Spill |
|---|---:|---:|
| `simt_fp32_pairwise_reg_scores` | 254 | 0 |
| `simt_fp32_pairwise_smem_scores` | 238 | 0 |
| `simt_fp32_pairwise_compact_smem_scores` | 253 | 0 |
| `tensor_core_tf32_wmma` | 166 | 0 |

Compact 版本的动态 shared memory 为 50,176 bytes（49 KiB）。RTX 3060
Laptop 的驱动还会为每个 block 保留 1,024 bytes，因此 NCU 中每 block 的
Shared Memory Configuration Size 应为 51,200 bytes（50 KiB）。两个 block
正好占用 102,400 bytes；是否实际驻留两个 block 应以 NCU 的 Block Limit 和
Theoretical Occupancy 为准。

## Python 调用

统一 dispatch 接口：

```python
import flash_attn_cuda as flash_attn

out = flash_attn.forward(
    q,
    k,
    v,
    implementation="simt_fp32_pairwise_reg_scores",
)

out = flash_attn.forward(
    q,
    k,
    v,
    implementation="simt_fp32_pairwise_smem_scores",
)

out = flash_attn.forward(
    q,
    k,
    v,
    implementation="simt_fp32_pairwise_compact_smem_scores",
)

out = flash_attn.forward(
    q,
    k,
    v,
    implementation="tensor_core_tf32_wmma",
)
```

默认 backend 是：

```text
simt_fp32_pairwise_smem_scores
```

也可以直接调用对应函数：

```python
out = flash_attn.simt_fp32_pairwise_reg_scores_forward(q, k, v)
out = flash_attn.simt_fp32_pairwise_smem_scores_forward(q, k, v)
out = flash_attn.tensor_core_tf32_wmma_forward(q, k, v)
```

## TF32 Tensor Core 基线

`tensor_core_tf32_wmma` 保持 FP32 BHND 输入/输出和固定 `head_dim=64`，
但将 Q/K/V 与 softmax probability 显式转换为 TF32：

```text
QK：WMMA m16n16k8，TF32 输入，FP32 accumulator
Softmax：FP32 CUDA Core
PV：WMMA m16n16k8，TF32 输入，FP32 accumulator
Output：FP32
```

每个 block 使用 4 个 warp 并处理 16 行 Query。动态 SMEM 为 45,248 bytes，
编译结果为 166 registers/thread、0 spill；SASS 已确认生成
`HMMA.1684.F32.TF32` 指令。该 backend 要求 Ampere（SM 8.0）或更新架构。
由于 TF32 mantissa 精度低于 FP32，正确性测试使用 `5e-3` 容差。

## 兼容旧名称

为了避免原有脚本立即失效，以下旧函数仍然保留：

```python
flash_attn.simt_register_tiled_forward(q, k, v)
flash_attn.simt_shared_scores_forward(q, k, v)
flash_attn.flash_attn_v2_gemm(q, k, v)
flash_attn.flash_attn_v2_gemm_kv_reuse(q, k, v)
```

旧 dispatch 字符串也仍然有效：

```python
flash_attn.forward(q, k, v, "simt_register_tiled")
flash_attn.forward(q, k, v, "simt_shared_scores")
```

新代码应使用规范化名称，旧名称只作为过渡兼容层。

## 编译

先激活安装了 PyTorch 与 CUDA 开发依赖的虚拟环境。下文的 `python` 指该环境中的
解释器，而不是系统 Python；可先确认：

```bash
which python
python -c "import torch; print(torch.__version__, torch.version.cuda)"
```

指定目标 GPU 的 compute capability：

```bash
TORCH_CUDA_ARCH_LIST=8.6 python setup.py build_ext --inplace
```

将 `8.6` 替换成目标 GPU 的 compute capability。构建默认开启：

```text
-O3
-lineinfo
--ptxas-options=-v
```

这样编译日志会显示 registers、spill、stack frame 等信息。基线构建暂不启用
`--use_fast_math`，避免一次 profiling 对比中同时改变算法和数学精度。

只有在单独研究 fast-math 影响时，才为 SGEMM extension 开启：

```bash
CUDA_KERNEL_LAB_SGEMM_FAST_MATH=1 \
TORCH_CUDA_ARCH_LIST=8.6 python setup.py build_ext --inplace
```

## 测试和 Benchmark

正确性测试：

```bash
python -m pytest -q tests
```

FlashAttention 性能测试：

```bash
python benchmarks/benchmark_forward.py \
  --seq-len 64 256 1024 \
  --repeats 5 \
  --output-json results/flash-attention/forward.json
```

SGEMM 性能测试：

```bash
python benchmarks/benchmark_sgemm.py \
  --m 4096 --n 4096 --k 4096 \
  --repeats 5 \
  --output-json results/sgemm/4096-cube.json
```

Python API：

```python
from cuda_kernel_lab import flash_attention, sgemm

c = sgemm.forward(a, b, "simt_fp32_03_transposed_smem_vec4")
o = flash_attention.forward(q, k, v, "simt_fp32_pairwise_smem_scores")
```

为 Nsight Compute 准备单次 kernel launch：

```bash
python benchmarks/profile_one.py \
  --implementation simt_fp32_pairwise_compact_smem_scores \
  --seq-len 1024 \
  --warmup 10
```

也可以直接使用 NCU 包装脚本：

```bash
./profile.sh flash simt_fp32_pairwise_compact_smem_scores 1024
```

SGEMM：

```bash
./profile.sh sgemm simt_fp32_03_transposed_smem_vec4 4096 4096 4096
```

详细实验契约见 [docs/experiments.md](docs/experiments.md)，profiling 方法见
[docs/profiling.md](docs/profiling.md)，实现演进路线见
[docs/architecture.md](docs/architecture.md)。

## 后续演进方向

SIMT 路线建议按独立 backend 继续添加：

```text
pairwise register scores
    ↓
pairwise shared-memory scores
    ↓
score recompute
    ↓
降低每线程 Q/O fragment
    ↓
K/V 双缓冲
    ↓
cp.async pipeline
```

Tensor Core 路线建议从独立实验开始：

```text
FP16/BF16 storage + FP32 scalar accumulation
    ↓
WMMA QK micro-kernel
    ↓
tiled QK + online softmax
    ↓
Tensor Core PV
    ↓
CUTLASS/CUTE 或显式 mma.sync pipeline
```

新优化不要覆盖已有 backend。应添加新的实现名称，复用同一测试集，然后与前一
阶段进行 profiling 对比。
