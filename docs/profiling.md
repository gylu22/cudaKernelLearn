# CUDA Profiling 工作流

本页是仓库 profiling 知识体系的总入口。原则是先确认测量可信，再解释指标；先从
应用缩小到 kernel，再从 kernel 缩小到源码和指令。

## 学习地图

```text
实验方法与可信基线
  ├── CPU wall / CUDA Event / NSYS / NCU 的计时边界
  ├── warmup、统计、控制变量、replay
  └── correctness 与 sanitizer
        ↓
应用级定位
  └── Nsight Systems：CPU、API、stream、memcpy、kernel、gap
        ↓
单 kernel 粗分类
  ├── Launch：grid、wave、register、SMEM、occupancy
  └── SOL：compute/memory 子系统吞吐
        ↓
单 kernel 根因
  ├── Memory：request、sector、wavefront、cache、spill
  ├── Compute：pipeline、instruction mix、MUFU/Tensor
  ├── Scheduler：active、eligible、issued、stall
  └── Roofline：工作量与上界
        ↓
源码归因
  └── CUDA source → PTX → SASS → producer/consumer
        ↓
单因素实验、shape 矩阵、回归
```

推荐阅读：

1. [Profiling 方法论](profiling/00-methodology.md)
2. [可复现实验与热点筛选](profiling/01-workflow.md)
3. [Nsight Systems](profiling/06-nsight-systems.md)
4. [Speed of Light](profiling/02-speed-of-light.md)
5. [Memory Workload](profiling/03-memory-workload.md)
6. [Compute、Scheduler 与 Stall](profiling/04-compute-and-scheduler.md)
7. [Occupancy](profiling/occupancy.md)与[Cache](profiling/cache.md)
8. [Roofline](profiling/05-roofline.md)
9. [NCU 指标字典](profiling/08-metric-reference.md)
10. [Source/PTX/SASS](profiling/07-source-and-sass.md)
11. [仓库案例](profiling/09-case-studies.md)
12. [Profiling 面试问答](profiling/10-interview-playbook.md)

## 1. 固定实验条件

至少记录：

- GPU 名称、compute capability、功耗/频率状态；
- driver、CUDA Toolkit、PyTorch 和 Nsight Compute 版本；
- git commit、backend、shape、dtype、warmup/iteration；
- `TORCH_CUDA_ARCH_LIST`、fast-math、寄存器限制等编译选项。

短 kernel 容易受初始化、DVFS 和 launch overhead 影响。先 warmup，使用 CUDA Event
计时，报告多次独立运行的中位数或分位数，不只保留最好一次。

## 2. 构建并记录编译器资源

```bash
TORCH_CUDA_ARCH_LIST=8.6 python setup.py build_ext --inplace
```

`-lineinfo` 让 profiler 将 SASS/PTX 映射回源码；`--ptxas-options=-v` 输出
register、spill、stack、constant memory。目标架构应替换为实际 GPU。

不要在一次对比中同时改变 tile、`--use_fast_math` 和数据类型。SGEMM 的
fast-math 实验使用独立开关：

```bash
CUDA_KERNEL_LAB_SGEMM_FAST_MATH=1 \
TORCH_CUDA_ARCH_LIST=8.6 python setup.py build_ext --inplace
```

## 3. Correctness 先于性能

```bash
python -m pytest -q
```

数值容差必须与实际数学路径匹配。TF32/FP16 不能沿用严格 FP32 容差；同时记录
max/mean error，避免只看一次 `assert_close`。

## 4. Benchmark

```bash
python benchmarks/benchmark_forward.py \
  --seq-len 64 256 1024 --repeats 5
python benchmarks/benchmark_sgemm.py \
  --m 4096 --n 4096 --k 4096 --repeats 5
```

端到端时间、Python 调度时间和单 kernel 时间回答不同问题。当前 benchmark 用
CUDA events 包围重复的 native extension 调用；输出创建仍发生在调用内，但结果
不是完整 CPU wall time，也不是剥离分配后的单 kernel duration。
统一计时和 JSON schema 见[实验基础设施](experiments.md)。

## 5. 先用 Nsight Systems 找热点

Nsight Systems 适合查看 CPU/GPU 时间线、launch gap、同步、memcpy 和多 stream
重叠。应用包含多个 kernel 时，应先在时间线上确定值得深入的 launch；不要用
Nsight Compute 的单个报告替代应用级热点分析。

```bash
mkdir -p artifacts/nsys artifacts/ncu

nsys profile --trace=cuda,nvtx,osrt \
  -o artifacts/nsys/flash_attention \
  python benchmarks/benchmark_forward.py --seq-len 1024
```

## 6. 用 Nsight Compute 分析一个受控 launch

FlashAttention：

```bash
ncu --set full --nvtx --nvtx-include "simt_fp32_pairwise_smem_scores/" \
  --launch-count 1 --force-overwrite \
  -o artifacts/ncu/fa_smem_n1024 \
  python benchmarks/profile_one.py \
    --implementation simt_fp32_pairwise_smem_scores \
    --seq-len 1024 --warmup 10
```

SGEMM：

```bash
ncu --set full --nvtx --nvtx-include "simt_fp32_03_transposed_smem_vec4/" \
  --launch-count 1 --force-overwrite \
  -o artifacts/ncu/sgemm_transposed_4096 \
  python benchmarks/profile_sgemm.py \
    --implementation simt_fp32_03_transposed_smem_vec4 \
    --m 4096 --n 4096 --k 4096 --warmup 10
```

NCU 可能通过 replay 多次执行同一 launch 来收集互不兼容的计数器。报告中的
Duration 应与非 profiler benchmark 交叉检查；`--set full` 开销大，日常迭代可
选择少量 section，最终再采完整报告。

## 7. 推荐诊断顺序

1. Duration 是否稳定、grid 是否足够大。
2. [SOL](profiling/02-speed-of-light.md)：哪个资源接近其持续吞吐上限。
3. [Memory](profiling/03-memory-workload.md)：请求是否合并、shared 是否冲突、
   是否 spill。
4. [Occupancy](profiling/occupancy.md) 与 scheduler：是否有足够 eligible warp。
5. 只有 issue efficiency 不足时再解释 [stall reasons](profiling/04-compute-and-scheduler.md)。
6. 用 [Roofline](profiling/05-roofline.md) 判断算法强度与硬件上限的距离。
7. 回到 Source/SASS，将主要指标关联到具体指令和源码行。

Stall 是“warp 下一条指令为何暂时不能 issue”的分类，不等于对应原因占了同样比例
的 wall time。优化是否成功最终仍以 duration、正确性和可重复性为准。

## 8. 结果保存

原始 `.ncu-rep`/`.nsys-rep` 放在被忽略的 `artifacts/`。可版本控制的摘要放
`results/`，至少包含实验元数据、关键指标、结论和原始报告文件名。

## 9. 每次 profiling 必须产出的结论

完成一次分析后，应能明确回答：

1. 目标 kernel 占 operator/应用多少时间；
2. 工作量、最少 bytes、grid/wave 是多少；
3. 主要限制来自吞吐、依赖、同步、并行度还是固定开销；
4. 根因位于哪段源码/哪类指令；
5. 下一项改动为什么会影响该根因；
6. 哪些 counters 应随改动变化；
7. correctness 和 duration 是否改善；
8. 结论适用于哪些 shape，在哪里退化。

如果只能说“某个百分比高”，说明分析还没有完成。

## 参考资料

- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)
- [Nsight Compute Compute Triage Guide](https://docs.nvidia.com/nsight-compute/ComputeTriage/)
- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/)
