# 0. Profiling 方法论：从现象到可验证结论

profiling 不是打开 Nsight Compute 后寻找最高的红色数字，而是一个实验科学过程：

```text
定义问题
→ 建立工作量与硬件模型
→ 获得可信基线
→ 缩小瓶颈范围
→ 提出可证伪假设
→ 只改变一个主要因素
→ correctness + duration + counters 共同验证
→ 在 shape 矩阵上确认适用域
```

## 1. 先写 performance question

不好的问题：

> 为什么这个 kernel 慢？

可分析的问题：

> 在 RTX 3060 Laptop、FP32 严格模式、M=N=K=4096、输出预分配的条件下，
> `simt_fp32_03_transposed_smem_vec4` 为什么只有 reference 的某个比例？

一个问题至少明确：

- GPU 与 compute capability；
- dtype/math mode；
- shape/layout；
- backend；
- 计时边界；
- reference；
- correctness 容差；
- 目标是 latency、throughput 还是端到端。

## 2. 建立静态模型

运行 profiler 前先计算：

```text
理论 FLOP/operation:
最少 global bytes:
实际 tile 重复读取:
grid/block/warp 数:
每 block shared memory:
每 thread accumulator:
barrier 次数:
tail 比例:
```

以 SGEMM `128×128×8` tile 为例：

```text
每 K tile FLOP = 2 × 128 × 128 × 8
A/B input bytes = (128×8 + 8×128) × 4
grid = ceil(M/128) × ceil(N/128)
K-loop iterations = K/8
```

这使 profiler 数字有参照：若 DRAM bytes 远高于模型，就去找重复读取、无效 sector、
spill 或额外输出初始化。

## 3. Correctness 是 profiling 前置条件

错误 kernel 可以非常快：

- 漏算 tail；
- 少执行部分 reduction；
- 未等待 async copy；
- 越界读取恰好没有 crash；
- race 在当前运行碰巧得到正确值；
- 使用了不同精度路径。

每个 profile target 先保存：

```text
max_abs_error
max_rel_error
mean_error
NaN/Inf count
reference 与 math mode
```

结构性改动后优先跑 memcheck/racecheck/synccheck，再讨论性能。

## 4. 测量层次

### 4.1 CPU wall time

回答完整调用从 host 看花多久。会包含 Python/C++ dispatch、allocation、launch、
同步等。异步 API 若未在正确边界同步，CPU 时间可能只测到提交。

### 4.2 CUDA Event

测同一 stream 上两个 event 之间的 GPU 时间。适合稳定 benchmark，但要说明范围是否
包含 allocation、memset、多个 kernel。

### 4.3 Nsight Systems duration

来自时间线，可确认 kernel、memcpy、gap、并发与 API 阻塞。适合应用级归因。

### 4.4 Nsight Compute duration

NCU 为收集 counter 可能 replay、patch 或改变 cache/时钟环境。它适合解释单 launch，
但 duration 要和非 profiler baseline 交叉验证。

## 5. 让 baseline 可复现

### Warmup

排除：

- CUDA context 初始化；
- module/JIT load；
- allocator 首次路径；
- cache/power state 初始变化；
- framework autotune。

### 重复和统计

- 在一个 sample 内重复多次；
- 做多个独立 sample；
- 报 median；
- 对服务 latency 报 P90/P95/P99；
- 同时保留 min/max 或离散程度；
- 不只取最好的一次。

### 环境

记录：

```text
GPU / driver / CUDA / framework / profiler
power limit / application clocks / temperature
display 或其他进程
git commit / build flags / target SM
random seed / input source
```

短 kernel 更容易受到 launch、频率和系统噪声影响。

## 6. 控制变量

一次实验只改变一个主要因素：

```text
baseline: transposed shared layout
experiment: only add B padding
```

不要同时改变：

- tile；
- dtype；
- fast math；
- vector width；
- block threads；
- pipeline stages。

否则即使变快，也不能知道原因。

## 7. 工具选择

| 问题 | 工具 |
| --- | --- |
| 应用热点、CPU/GPU gap、并发 | Nsight Systems |
| 单 kernel 资源/吞吐/stall | Nsight Compute |
| 编译资源 | `ptxas -v` |
| 指令 | NCU Source、`cuobjdump`、`nvdisasm` |
| 内存错误 | Compute Sanitizer memcheck |
| shared race | racecheck |
| 同步原语问题 | synccheck |
| framework operator | PyTorch profiler（再下钻 NSYS/NCU） |
| microbenchmark | CUDA Event + 独立重复 |

工具选择顺序通常是从粗到细：

```text
application → operator → kernel → source line → instruction
```

## 8. NCU replay 与测量扰动

硬件不能一次收集所有 counter。NCU 会将请求的 metrics 分成多个 pass，并可能 replay
目标 kernel。

Kernel Replay 大致会：

1. 第一次运行识别 kernel 可访问/写入的内存；
2. 保存需要恢复的内容；
3. 在后续 pass 前恢复；
4. 多次执行相同 launch 收集不同 counters。

风险：

- 非确定 kernel 每 pass 工作不同；
- global atomic/队列会改变执行；
- cache 状态与正常运行不同；
- 极短 kernel 指标误差大；
- `--set full` pass 多、运行慢；
- display/copy engine 影响全芯片 L2/DRAM counters。

实践：

- 日常只采必要 section；
- 先保存非 NCU baseline；
- 确保输入/输出可 replay；
- 对状态依赖 kernel 考虑 application replay 或单独 harness；
- 对异常指标减少 metric 集合并重采；
- kernel 尽量超过极短的非稳态区间。

## 9. 指标层次

NCU 中常见数量关系：

```text
source expression
  → SASS instruction
      → request
          → sectors
              → wavefronts
                  → L1/L2/DRAM physical traffic
```

- instruction：实际执行的机器指令；
- request：向某硬件单元提交的工作；
- sector：常见为对齐 32 B 数据块；
- wavefront：硬件一次能够处理的 request 工作包；
- transaction/bytes：经过某层级的实际传输。

一条 memory instruction 可产生多个 request/sector/wavefront。只看源码 load 次数
不足以估算流量。

## 10. 诊断主链

### 第一步：Duration 与 launch

- kernel 是不是目标热点；
- grid/block 是否符合预期；
- block 数能否填满 GPU；
- duration 是否稳定；
- tail wave 是否明显；
- 是否混入 memset/copy。

### 第二步：SOL/Throughput 粗分类

- compute 哪条 pipeline 高；
- memory 是 DRAM、L2、L1/TEX 还是 shared 高；
- 两边都低时不要强行判 compute/memory bound。

### 第三步：资源与 scheduler

- registers/thread、SMEM/block；
- theoretical/achieved occupancy；
- active/eligible/issued warp；
- issue slot 是否空闲。

### 第四步：对应子系统

- memory：requested bytes、sector、wavefront、hit/miss、spill；
- compute：instruction mix、目标 pipeline、依赖；
- synchronization：barrier、async wait、负载不均；
- control：divergence、predication、tail。

### 第五步：Source/SASS

把 counter 归因到：

- 哪条 source；
- 哪条消费者在 stall；
- 对应 producer load 在哪里；
- 指令宽度和地址计算是否符合预期；
- 是否有意外 local load/store。

## 11. 从相关性到因果

错误推理：

> padding 后 bank conflict 降了，所以一定更快。

完整验证：

1. padding 改变实际 bank mapping；
2. shared excessive wavefront 减少；
3. 没有显著增加 shared footprint/register/address overhead；
4. resident blocks 没下降；
5. 目标 shape duration 稳定下降；
6. correctness 不变；
7. 多个相关 shape 重复成立。

counter 变化只是因果链中的证据，不是最终目标。

## 12. 如何写实验结论

```text
问题：
基线：
静态模型：
观察：
假设：
改动：
正确性：
duration：
关键 counters：
结论：
适用 shape：
反例/风险：
下一步：
```

不确定时写“证据支持”“更可能”，不要把相关性写成硬件事实。

## 13. 常见反模式

- 看到 Memory SOL 高就说 DRAM-bound；
- 看到 long scoreboard 就说带宽不足；
- 看到 occupancy 低就限制寄存器；
- 看到 cache hit 低就追求 hit rate；
- 看到 barrier stall 就删除必要 barrier；
- 看到某 stall 最大就认为它占同等 wall time；
- 用 profiler 下的 duration 做唯一性能数字；
- 只测一个方阵；
- 修改多个因素后给单一因素归功；
- 忽略数值路径变化。

## 14. 参考资料

- [Nsight Compute Profiling Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)
- [Nsight Compute Compute Triage](https://docs.nvidia.com/nsight-compute/ComputeTriage/)
- [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)

