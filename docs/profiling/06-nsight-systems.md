# 6. Nsight Systems：从应用时间线定位真正热点

Nsight Systems（NSYS）回答“时间花在哪里、任务是否重叠、CPU 为什么没持续向 GPU
供给工作”。它不负责解释单个 kernel 内某条 shared load 为什么有 bank conflict；
那是 Nsight Compute 的任务。

## 1. 什么时候先用 NSYS

- 应用包含多个 kernel；
- operator 有 allocation、memset、copy；
- CPU launch gap 明显；
- 多 stream/通信/数据加载；
- CUDA Graph；
- cuBLAS/cuDNN 与自定义 kernel 混合；
- 单 kernel 优化后端到端没有改善。

如果不知道哪个 kernel 是热点，不应直接采整个应用的 NCU full set。

## 2. 基本采集

```bash
mkdir -p artifacts/nsys

nsys profile \
  --trace=cuda,nvtx,osrt \
  --sample=none \
  --force-overwrite=true \
  -o artifacts/nsys/sgemm \
  python benchmarks/benchmark_sgemm.py \
    --m 4096 --n 4096 --k 4096
```

是否启用 CPU sampling、context switch、GPU metrics 会影响开销和权限要求。先从最少
trace 开始，需要时再增加。

查看摘要：

```bash
nsys stats artifacts/nsys/sgemm.nsys-rep
```

## 3. NVTX

NVTX 用命名 range 将上层语义投影到 CPU/GPU 时间线：

```text
iteration
  ├── allocate
  ├── custom_sgemm
  ├── reference
  └── validation
```

好处：

- 过滤 warmup；
- 关联 Python/C++ operator 与 kernel；
- 区分同名 kernel 的不同 backend/shape；
- 为 NCU `--nvtx-include` 提供精确目标；
- 观察 graph capture/replay 范围。

range 名建议包含 backend 和关键 shape，但不要每个 lane/小步骤都产生 host 标记。

## 4. 时间线阅读顺序

### 4.1 CPU thread

查看：

- 哪个线程提交 CUDA；
- API 调用是否阻塞；
- `cudaDeviceSynchronize`/event synchronize；
- allocation/free；
- Python 或 framework gap；
- mutex/OS runtime。

### 4.2 CUDA API

区分：

- API host duration；
- 对应 GPU activity duration；
- 异步调用是否立即返回；
- API 之间是否有长 gap；
- 隐式同步来自哪里。

### 4.3 GPU stream

查看：

- kernel 顺序；
- memcpy/memset；
- stream 间 overlap；
- kernel 间 bubble；
- 多个小 kernel；
- 通信与计算是否重叠；
- graph launch 是否减少 gap。

### 4.4 GPU 全局利用

时间线上“有 kernel 在跑”不等于整块 GPU 饱和。小 grid 可能只用少量 SM。NSYS
定位时间区间后，仍需 NCU 判断单 kernel 资源利用。

## 5. 常见时间线模式

### 模式 A：短 kernel + 大 launch gap

```text
CPU: launch ---- gap ---- launch ---- gap
GPU: kernel      idle    kernel      idle
```

方向：

- fusion；
- batching；
- CUDA Graph；
- 减少 Python/C++ dispatch；
- 异步提交。

### 模式 B：H2D 与 kernel 串行

```text
H2D → kernel → H2D → kernel
```

检查 pinned memory、不同 stream、event 依赖、copy engine 和 buffer 双缓冲。

### 模式 C：allocation/memset 占比高

自定义算子内部 `zeros` 可能产生额外 memset。SGEMM 当前 binding 就存在需要进一步
拆分验证的输出清零成本。

### 模式 D：多个 stream 仍无 overlap

可能因为：

- 默认 stream 依赖；
- event 链；
- 每个 kernel 已占满 SM/寄存器/SMEM；
- 共享 DRAM 带宽；
- host 提交不及时。

### 模式 E：单个长 kernel

应用级热点已明确，转 NCU 下钻。不要在 NSYS 时间线猜 bank conflict。

## 6. CPU/GPU 时间的常见误读

- CUDA API 10 µs 不代表 GPU kernel 10 µs；
- kernel launch API 返回不代表 kernel 完成；
- `cudaMemcpy` 是否阻塞取决于方向、内存类型和 API；
- GPU stream 中的 gap 可能来自 CPU 未提交，也可能来自依赖；
- overlap 不能只看矩形视觉重叠，要确认资源与时间轴；
- profiler 本身会产生开销。

## 7. 与 NCU 协作

推荐流程：

1. NSYS 找占总 GPU 时间最多的 kernel；
2. 记录 kernel 名、调用位置、shape 和 NVTX range；
3. 写单 launch profiling harness；
4. 用 event 重现该 duration；
5. NCU 只采目标 launch；
6. 优化后重新回到 NSYS 看端到端收益。

一个 kernel 自身快了，但因增加预处理、memset 或 layout conversion，operator 可能
反而更慢。

## 8. 本仓库建议

FlashAttention：

- 一个 backend 是否只 launch 一次；
- 输出分配是否显著；
- 不同序列长度下 launch 与 kernel 比例；
- 与 PyTorch SDPA 的 kernel 数和融合边界。

SGEMM：

- `torch::zeros` 是否产生 memset；
- 自定义 operator 与 `torch.matmul` 计时边界；
- 小 K 时 launch/memset 占比；
- 多 shape sweep 中是否频繁同步。

## 9. 面试问题

### NSYS 和 NCU 的区别？

NSYS 是系统级时间线，定位 CPU、CUDA API、memcpy、kernel、stream 与通信；NCU
分析单 kernel 的硬件 counter、资源、scheduler、memory 和 source/SASS。

### Kernel 更快但应用没变快怎么办？

回到 NSYS 检查热点占比、launch gap、allocation、copy、同步以及是否新增前后处理，
再用 Amdahl 定律解释端到端上限。

## 10. 参考资料

- [Nsight Systems User Guide](https://docs.nvidia.com/nsight-systems/UserGuide/)
- [NVTX Documentation](https://docs.nvidia.com/nvtx/)

