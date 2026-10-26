# 2. Speed of Light 吞吐分析

![Speed of Light 示例](assets/speed-of-light.png)

SOL section 用于回答“哪些硬件资源接近其可达到的吞吐上限”。它不会直接告诉你
根因，也不能单独区分带宽受限和延迟受限。

## Compute 与 Memory

- **Compute (SM)** 聚合一个或多个 SM 执行管线的吞吐；具体管线要到 Compute
  Workload Analysis 查看。
- **Memory** 聚合内存系统相关资源；要继续区分 L1/TEX、L2、DRAM 和 shared
  memory。
- 聚合百分比可能由某个子单元的高利用率主导，因此必须向下展开。

“SM SOL 高”不等于所有 CUDA Core 都忙，也不等于算法已达理论 FLOPS；“Memory
SOL 高”也不一定是 DRAM 饱和，可能是 L1/TEX 或 shared 路径受压。

## 分层诊断

### DRAM 高

检查：

- 实际 DRAM bytes 与算法最低必要流量；
- global load/store 是否合并；
- 是否存在无复用的中间 tensor；
- arithmetic intensity 是否天然偏低。

### L2 高而 DRAM 较低

说明 L2 路径繁忙，但数据可能命中 L2。确认 sector 数、读写方向和跨 block 复用；
L2 高不是自动的坏事。

### L1/TEX 高而 DRAM/L2 不高

可能来自 shared-memory 指令、global/local load-store 或 texture/surface 请求，因为
现代架构的 L1/TEX 单元服务多种路径。结合 Memory Workload 和 short-scoreboard/
MIO throttle 判断，不能仅凭 L1 百分比断言“缓存未命中”。

### Compute 高

展开 pipeline：FP32、FP64、Tensor、INT、MUFU、LSU 等。某条管线接近上限才说明
对应指令吞吐可能限制性能；随后用 Source/SASS 确认是哪段代码产生这些指令。

## 吞吐受限与延迟受限

| 现象 | 更可能的方向 |
|---|---|
| 资源吞吐高、issue 稳定 | 吞吐受限；减少工作或换更合适的数据路径 |
| SOL 低、eligible warp 低 | 延迟/依赖/同步；查看 scheduler 和 stall |
| occupancy 高但 issue 低 | 有 warp 驻留但未 ready；增加 ILP 或修复依赖 |
| occupancy 低但吞吐高 | 当前 warp 已足够；继续提高 occupancy 未必有益 |

最终用 duration 验证。降低某个 SOL 百分比但让 kernel 更慢不算优化。

## 典型组合的诊断

| Compute | Memory | Scheduler | 更可能的方向 |
| --- | --- | --- | --- |
| 高 | 低/中 | issue 稳定 | 展开目标计算 pipeline，减少算术或换指令 |
| 低/中 | 高 | issue 稳定 | 展开 DRAM/L2/L1/shared，检查流量效率 |
| 高 | 高 | issue 稳定 | 多资源同时活跃，比较哪项改动最影响 duration |
| 低 | 低 | eligible 低 | latency、barrier、依赖、grid/tail |
| 低 | 低 | eligible 高 | dispatch/某个未被聚合突出的 pipeline |
| 低 | 低 | grid 很小 | 并行度/launch，而非单 SM 峰值 |

表格只用于选择下一页，不是最终瓶颈判定。

## 按层级提问

### Compute 高时

1. 高的是 FP32、FP64、Tensor、INT、MUFU 还是 LSU？
2. 这些指令是必要工作还是地址/转换/recompute？
3. active lane 是否足够？
4. scheduler 是否持续供给该管线？
5. 能否减少 operation、融合或改 dtype？

### Memory 高时

1. 是 DRAM、L2、L1/TEX 还是 shared？
2. physical bytes 是否接近算法最低 bytes？
3. sectors/request 与 wavefront 是否合理？
4. cache hit 是否符合预期复用？
5. 是否来自 local spill 或中间 tensor？

## SOL 百分比下降的两种相反含义

```text
优化后 duration 下降、工作减少
→ 相同硬件峰值下利用率可能下降，这是成功

优化后 duration 上升、供给不足
→ 利用率下降是退化
```

所以 SOL 从来不能离开 workload 与 duration 单独比较。

## 参考资料

- [Nsight Compute Profiling Guide: Speed of Light](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)
- [Nsight Compute Compute Triage Guide](https://docs.nvidia.com/nsight-compute/ComputeTriage/)
