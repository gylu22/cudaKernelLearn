# 5. Roofline 模型

![Roofline 示例](assets/roofline.png)

Roofline 将计算吞吐、内存带宽和 arithmetic intensity 放到一张对数坐标图中：

```text
Arithmetic Intensity = 浮点操作数 / 该层级传输字节数
Attainable Performance <= min(Compute Peak, Bandwidth × Arithmetic Intensity)
```

## 读图

- X 轴：FLOP/byte；
- Y 轴：FLOP/s；
- 斜线：给定内存层级的带宽上限；
- 水平线：所选数学路径的计算上限；
- ridge point：两条边界交点。

点在 ridge 左侧通常更受数据供给约束，右侧通常更受计算上限约束。点离对应 roof
很远说明仍有优化空间，但原因可能是依赖、同步、工作不均、低 occupancy、指令
混合或测量口径不匹配，不能只称为“延迟受限”。

## 先确认口径

1. FLOP 统计是否覆盖当前指令路径：普通 FP32、Tensor Core、特殊函数不能混为一谈。
2. byte 是 DRAM、L2 还是 L1 层级；不同层级会得到不同 arithmetic intensity。
3. FMA 通常按两个 FLOP 计，但工具 section 的定义应作为准绳。
4. 问题规模是否足够大，频率和带宽是否稳定。

NCU 支持 hierarchical roofline；Tensor Core kernel 应使用匹配的 tensor roofline
section，并与 SASS/MMA 指令计数交叉验证。

## 优化方向

### 接近带宽 roof

- 减少中间结果的 materialization；
- 提高寄存器/shared/cache 复用；
- 合并访问并减少无效 sector；
- 融合 kernel，前提是资源使用不会抵消收益。

### 接近计算 roof

- 减少冗余操作；
- 使用适合精度目标的 Tensor Core/低精度路径；
- 改善指令组合和依赖链；
- 检查数学函数是否占用稀缺管线。

### 远离 roof

结合 scheduler、stall、occupancy、tail wave 和 load balance。高“物理带宽”与低
有效 FLOP/s 可以来自无效 sector，但必须由 memory counters 证明，不能只看图猜测。

## 手工计算示例

假设一个 FP32 vector add：

```text
C[i] = A[i] + B[i]
```

每元素约：

- 1 FLOP；
- 读取 A/B 共 8 B；
- 写 C 4 B；
- 理想 DRAM AI 约 `1/12 FLOP/B`。

它天然处于低 AI 区域。通过循环展开可以减少地址/控制指令，但无法把算法变成高 AI；
主要方向是合并访问、减少额外流量或与前后算子融合。

对 tiled SGEMM，每个 A/B tile 在 block 内重复参与许多 FMA，AI 显著提高。因此 tile
设计和复用决定它能否向 compute roof 移动。

## Hierarchical Roofline

同一个 kernel 可能：

- 相对 DRAM 有很高 AI，因为数据被 L2 复用；
- 相对 L1/shared 有较低 AI，因为片上搬运频繁；
- 达到 DRAM roof 却受 L1/shared 子系统继续限制。

因此层次化 roofline 应与 Memory Workload 的各级 bytes 对齐。不能用 DRAM bytes
计算 AI，却拿 L2 bandwidth 斜线解释。

## Roofline 实验

改变 shape 或 tile，观察：

```text
FLOP 是否不变
DRAM/L2/L1 bytes 如何变化
AI 是否移动
achieved FLOP/s 是否沿斜线移动
是否碰到新的 horizontal roof
```

若点没有按模型移动，检查 operation/byte 口径、cache、tail 和 NCU section。

## 参考资料

- [Nsight Compute Profiling Guide: Roofline Charts](https://docs.nvidia.com/nsight-compute/ProfilingGuide/#roofline-charts)
- [Nsight Compute Compute Triage Guide](https://docs.nvidia.com/nsight-compute/ComputeTriage/)
