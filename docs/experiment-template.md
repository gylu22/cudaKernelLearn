# CUDA 实验记录模板

复制本页到 `results/<topic>/<date>-<experiment>.md`。原始 profiler 文件放到
被忽略的 `artifacts/<topic>/`，并在摘要中记录文件名。性能结论必须能追溯到环境、
输入和命令。

## 目标与假设

- 要回答的问题：
- 预期瓶颈：
- 本次只改变的变量：
- 成功标准：

## 环境

| 项目 | 值 |
| --- | --- |
| Git commit / dirty state | |
| GPU / compute capability | |
| Driver / CUDA toolkit | |
| PyTorch / compiler | |
| GPU clocks / power mode | |
| 构建参数 / `TORCH_CUDA_ARCH_LIST` | |

## 工作负载

| 参数 | 值 |
| --- | --- |
| Kernel / backend | |
| Shape / layout / dtype | |
| warmup / iterations | |
| stream / concurrency | |
| 输入数据分布 | |

## 正确性

- 参考实现与容差：
- 最大绝对/相对误差：
- 边界 shape：
- 测试命令：

## 性能

| 指标 | baseline | candidate | 变化 |
| --- | ---: | ---: | ---: |
| Median latency | | | |
| P90 / P99 latency | | | |
| Throughput / TFLOP/s | | | |
| Effective bandwidth | | | |

```bash
# benchmark / nsys / ncu 命令
```

记录 duration、Compute/Memory SOL、DRAM/L1/L2、eligible/issued warps、occupancy 限制因素，以及确实阻止 scheduler issue 时的主要 stall reason。

## 结论

- 假设是否成立：
- 改动为什么有效或无效：
- 结论适用的 shape / GPU 范围：
- 下一步实验：
