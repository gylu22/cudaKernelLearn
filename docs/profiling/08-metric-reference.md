# 8. Nsight Compute 指标关系与诊断字典

本页不是让你背完整 metric 名，而是建立“问题—指标组—解释—下一步”的关系。不同
NCU 版本和 GPU 架构的 exact metric 名会变化；优先使用 section 展示的语义，再通过
CLI 查询当前设备支持的 metric。

```bash
ncu --list-sections
ncu --query-metrics
ncu --query-metrics --query-metrics-mode suffix
```

## 1. Metric 名的基本结构

常见 raw metric 形如：

```text
unit__signal.qualifier
```

示例单位前缀：

- `sm__`：SM 聚合；
- `smsp__`：SM sub-partition；
- `l1tex__`：L1/TEX/shared 相关子系统；
- `lts__`：L2 slice；
- `dram__`：device memory；
- `launch__`：launch/资源模型；
- `gpu__`：GPU 范围。

常见 suffix 可能表示：

- `.sum`：所有实例/周期求和；
- `.avg`：实例平均；
- `.max/.min`；
- `.per_cycle_*`；
- `.pct_of_peak_sustained_*`；
- ratio/percentage。

同一个 raw counter 使用不同 suffix 会回答不同问题。不要把 `.sum` 与 `.avg` 直接
跨 GPU 比较。

## 2. Duration 与频率

先记录：

- GPU duration；
- elapsed cycles；
- SM/DRAM frequency；
- 多次采样的稳定性。

同样 cycles 在不同频率下时间不同；同样时间也可能因 boost 状态不同产生不同
counters。对短 kernel，固定开销和测量误差更显著。

## 3. Launch Statistics

关注：

```text
grid dimensions
block dimensions
threads/block
registers/thread
static/dynamic shared memory
waves per SM
theoretical occupancy
block limit reason
```

### Waves per SM

它帮助判断 grid 规模与 wave tail。很少的 waves 常意味着：

- 小问题；
- tile 太大；
- shape 瘦长；
- block residency 低。

waves 很多不代表均衡，最后 wave 和每 block 工作差异仍需考虑。

### Block limit

若限制来自：

- register：尝试缩短 live range/调整 thread tile，但防 spill；
- shared memory：减少 stage/tile，但防流量上升；
- warps/threads：检查 block size；
- blocks：小 block 可能碰架构 block 数上限。

只在 scheduler 确实缺少 eligible warp 时，提升 residency 才更可能有效。

## 4. Speed of Light

SOL 用于粗分：

```text
Compute Throughput
Memory Throughput
```

必须向下展开：

- Compute：FP32、FP64、Tensor、INT、MUFU、LSU 等；
- Memory：DRAM、L2、L1/TEX、shared。

错误结论：

```text
Memory SOL 高 = DRAM bandwidth bound
Compute SOL 高 = FP32 compute bound
```

聚合值可能由任意子单元主导。

## 5. Scheduler Statistics

核心关系：

```text
theoretical warps
  ≥ active warps
      ≥ eligible warps
          → issued warps
```

### Active 高、eligible 低

warp 驻留但等待：

- memory dependency；
- execution dependency；
- barrier；
- instruction fetch；
- async wait。

### Eligible 高、issued 受限

更像：

- 目标 pipeline/dispatch 吞吐限制；
- scheduler issue 能力；
- 多个 ready warp 竞争相同资源。

### Active 低但吞吐高

当前 ILP/TLP 已足以饱和目标资源。提高 occupancy 可能无益。

## 6. Warp stall reason

只在 issue slot 经常空闲时优先分析。

| 状态 | 常见含义 | 必须交叉检查 |
| --- | --- | --- |
| long scoreboard | 等待较长 memory dependency | global/local sector、producer load |
| short scoreboard | 等待 MIO/shared/特殊路径依赖 | shared wavefront、MUFU、source |
| wait | 固定 latency execution dependency | 指令链、ILP、unroll |
| barrier | 等 CTA barrier 的其他 warp | barrier 次数、work imbalance |
| branch resolving | 控制流解析 | branch/source/active lanes |
| MIO throttle | MIO 队列压力 | shared/special/memory 指令数 |
| LG throttle | local/global 指令队列压力 | global/local requests |
| math pipe throttle | 数学管线繁忙 | pipeline utilization |
| not selected | ready 但选了别的 warp | 通常说明有可调度工作 |
| drain | 退出前等未完成操作 | 尾部 store、tail effect |

名称和归类随架构变化。stall 是采样状态，不是 wall time 百分比。

## 7. Global/L1/L2/DRAM

### Request

硬件单元收到的一次工作。一个 warp instruction 可能生成一个或多个 request。

### Sector

常见为 32 B 对齐块。L1/L2 128 B cache line 可由 4 个 sector 组成。

### Sectors/request

在 active lane 数相近时，更少通常说明地址覆盖更紧凑。但：

- 每 lane 访问宽度不同；
- broadcast/reuse；
- tail active lane 不同；
- L1 与 L2 request 拆分规则不同。

因此只对可比指令/shape 使用。

### Hit rate

必须与：

- request/sector 总量；
- sector misses to next level；
- DRAM bytes；
- 算法预期复用；
- cold/warm 状态

一起解释。

### Requested vs actual

```text
useful bytes：算法真正需要
requested bytes：指令请求
transferred bytes：sector/cache line/DRAM 实际传输
```

三者差距指出重复工作或传输效率，但 cache 可能让同一 physical sector 服务多个
request。

## 8. Shared memory

查看：

- shared load/store instructions；
- requests；
- wavefronts；
- ideal wavefronts；
- excessive wavefronts；
- throughput；
- short scoreboard/MIO throttle。

bank conflict 的更直接证据是额外 wavefront，而不只是源码 stride。向量指令本身也
可能需要多个正常 wavefront，因此要比较 ideal 与 actual。

## 9. Local memory 与 spill

结合：

```text
ptxas spill loads/stores
local load/store instructions
local sectors
L1/L2 traffic
registers/thread
source/SASS LDL/STL
```

`ptxas` 报 0 spill 不代表绝无 local memory：动态索引私有数组、栈等也可能使用
local address space。

## 10. Compute pipeline

关注：

- 目标 pipeline `% peak sustained`；
- executed instruction mix；
- cycles active；
- scheduler issue；
- dependent instruction spacing；
- active lanes；
- conversion/address overhead。

### FP32 高

可能真正 compute-throughput limited；也可能许多 FP32 是可避免的 recompute。

### Tensor 高但整体不快

检查：

- load/layout conversion；
- shared staging；
- softmax/epilogue；
- tensor instructions 占整个 kernel 比例；
- tile 利用率；
- pipeline feed 是否有 gap。

### MUFU 高

exp/reciprocal/sqrt 可能主导。减少调用、使用更合适近似或重排算法前先确认数值契约。

## 11. Branch 与 predication

查看：

- branch efficiency；
- divergent branch；
- active threads per warp/instruction；
- predicated-on/off threads；
- tail 比例；
- source/SASS。

高 branch efficiency 不代表无浪费：大量 predicated-off 指令仍会消耗发射。低值也
可能只出现在极短边界路径，对总时间影响小。

## 12. Roofline

确认：

- FLOP 统计对应 FP32/Tensor/其他路径；
- FMA 计两个 FLOP；
- bytes 对应 DRAM/L2/L1 哪一级；
- kernel 足够稳定；
- 频率合理；
- operation count 没漏 softmax/转换。

点离 roof 远时，转 scheduler、occupancy、tail、同步和 source，而不是简单宣布
“优化空间很大”。

## 13. Metrics 的单位陷阱

- cycle 是哪个 unit 的 cycle；
- rate 是 per SM、per sub-partition 还是全 GPU；
- pct of peak 是 active cycles 还是 elapsed cycles；
- average 是对实例平均还是对时间平均；
- hit rate 的 numerator/denominator 是否跨 pass；
- duration 是否包含 replay/patch；
- 聚合 sum 跨不同 SM 数不能直接比较。

NCU UI 中打开 metric details，CLI 使用
`ncu --query-metrics --query-metrics-mode suffix` 查精确定义。

## 14. 最小指标集

日常 triage 不必每次 `--set full`：

```text
LaunchStats
SpeedOfLight
Occupancy
SchedulerStats
WarpStateStats
MemoryWorkloadAnalysis
ComputeWorkloadAnalysis
SourceCounters（需要下钻时）
```

先用少量 section 建立方向，再针对 memory/source/roofline 采集。

## 15. 参考资料

- [Nsight Compute Profiling Guide: Metrics Guide](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)
- [Nsight Compute CLI](https://docs.nvidia.com/nsight-compute/NsightComputeCli/)
