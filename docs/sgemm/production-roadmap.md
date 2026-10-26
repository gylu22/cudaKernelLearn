# SGEMM 生产化差距与后续路线

当前 SGEMM 已经是很好的 CUDA 学习实现：它展示了 shared-memory tiling、thread tiling、向量化、转置、双缓冲结构和 padding，并且在适合固定 tile 的 shape 上能够接近 cuBLAS。

但“某些 shape 达到 cuBLAS 95%+”只能证明一个 kernel 在该性能区域内有效，还不能代表生产可用。生产级实现需要同时处理正确性覆盖、API 语义、性能稳定性、硬件兼容、可诊断性和长期回归。

当前性能差距的具体原因见 [SGEMM 优化思路与性能差距分析](optimization-guide.md)。

## 1. 当前实现与生产要求的差距

| 维度 | 当前状态 | 生产级要求 |
| --- | --- | --- |
| shape | 快 kernel 要求 M/N 被 128 整除、K 被 8 整除 | 任意合法 shape，无越界且无性能断崖 |
| dtype | FP32 SIMT | 至少明确 FP32、TF32、FP16、BF16 的支持和累加策略 |
| layout | rank-2 contiguous row-major | stride、转置、常见 layout，或明确可靠 fallback |
| API | 固定 `C=A@B` | `alpha/beta`、out、batch、transpose，必要时 epilogue |
| dispatch | 用户显式选 backend | 依据 shape/dtype/架构自动选 kernel |
| fallback | 不满足条件时报错 | 回退到通用 kernel 或 cuBLAS |
| pipeline | 同步 load；结构上的双缓冲 | 真正异步、多 stage、可按架构选择 |
| 并行策略 | 固定 `128×128×8` | tile family；必要时 split-K/Stream-K/persistent |
| 数值 | 基础误差测试 | dtype 分级容差、NaN/Inf、确定性和累加语义 |
| 安全性 | launch check 与单元测试 | sanitizer、越界/race、随机 shape、长稳测试 |
| 运行环境 | 当前 GPU 上验证 | 多架构编译、兼容矩阵、多 GPU/device/stream |
| 性能保障 | 若干 benchmark 点 | workload 矩阵、回归阈值、结果可追溯 |
| 框架集成 | forward 扩展 | autograd 策略、CUDA Graph、编译/打包兼容 |

## 2. P0：先让性能数据可信

这是所有优化之前必须完成的工作。

### 2.1 去掉不必要的输出清零

当前 binding 为输出使用 `torch::zeros`，原因是 padded kernel 的 epilogue 会读取 C，即使 `beta=0`。应将纯 `C=A×B` 路径改为：

- 不读取旧 C；
- 使用 `torch::empty` 分配输出；
- 将 `beta != 0` 作为另一条显式路径。

否则小 K 和小矩阵的测量会混入一遍额外的 `M×N` 初始化。

### 2.2 分开测量三类时间

需要分别报告：

1. 纯 kernel：预分配输入输出，只测 launch 后的 GPU 工作；
2. operator：包含输出 tensor 分配；
3. 端到端：包含上层调用、必要同步和数据准备。

当前学习阶段最重要的是第一类，但生产决策不能只看第一类。

### 2.3 固定比较契约

每条 benchmark 结果必须记录：

- GPU 型号、SM 数、CUDA/driver/PyTorch 版本；
- dtype、TF32 设置、累加类型；
- layout、transpose、M/N/K；
- warmup、重复次数、统计量；
- cuBLAS 比较入口；
- 时钟或功耗状态是否稳定；
- 数值误差。

完成标准：同一提交重复运行的关键 shape 结果稳定，且 benchmark 不给任何一方附加不对称工作。

## 3. P1：覆盖一般 shape 与正确性边界

### 3.1 添加 tail-safe 路径

优先保留完整 tile 的无分支 fast path，再处理边界：

- interior block 继续使用向量化完整 tile；
- edge block 使用 predicated load/store，或单独的 edge kernel；
- 对不满足 `float4` 对齐的地址提供标量/较小向量 fallback；
- 覆盖 `M/N/K` 在 tile 边界前后的测试。

### 3.2 明确索引与边界语义

需要决定并测试：

- 空维度是否合法；
- 极大矩阵是否需要 64-bit 索引；
- NaN/Inf 如何传播；
- 输入别名和 output overlap 是否允许；
- 非 contiguous tensor 是直接支持、复制，还是回退。

### 3.3 建立随机与工具化测试

除固定样例外，应增加：

- 随机 M/N/K；
- 非对齐和质数维度；
- 大小值混合、NaN/Inf；
- `compute-sanitizer` memcheck；
- racecheck；
- 长循环压力测试。

完成标准：所有声明支持的 shape 都有定义清晰的结果，不依赖调用者绕开边界。

## 4. P2：建立可用的算子接口

生产接口至少应明确以下能力是否支持：

- `A`、`B` 转置；
- leading dimension 和 stride；
- batched/strided-batched GEMM；
- `C = alpha × A × B + beta × C`；
- out tensor；
- bias、activation 等融合 epilogue；
- autograd：由已有 PyTorch 算子组合，还是自定义 backward。

不必一次实现所有能力，但每个未支持分支都应有确定的 fallback，而不是只抛出与 tile 细节相关的错误。

还应验证：

- 当前 CUDA stream；
- 多 GPU 下正确的 device guard；
- CUDA Graph capture；
- 异步错误能在合适边界被发现；
- 扩展编译和 wheel 安装。

## 5. P3：从单 kernel 发展为 kernel family

固定 `128 × 128 × 8` 不可能覆盖所有 shape。建议先建立少量可解释的候选：

| 工作区间 | 候选 tile |
| --- | --- |
| 大型方阵 | `128×128` |
| N 较窄 | `128×64` 或 `64×64` |
| M 较窄 | `64×128` 或 `64×64` |
| 小矩阵/小 K | 更小 block tile 和 thread tile |
| 大 K、block 少 | split-K/Stream-K |

每个候选同时 sweep：

- `BK = 8/16/32`；
- thread tile；
- pipeline stage 数；
- 向量宽度；
- shared-memory layout。

先用离线 benchmark 生成决策表，再实现简单、可复现的 dispatcher。不要在缺少数据时堆叠大量 heuristic。

完成标准：目标 shape 矩阵中不存在由错误 tile 选择导致的明显性能悬崖，不适用区域能够可靠回退。

## 6. P4：实现真正的异步流水

在 Ampere 及之后架构上，下一步应验证 `cp.async`：

1. 建立同步单 buffer 基线；
2. 建立同步 double buffer 对照；
3. 实现 2-stage `cp.async`；
4. 再测试 3/4-stage 是否值得额外 shared memory；
5. 用 stall reason 和 eligible warps 证明 latency overlap，而不是只凭代码结构命名。

需要同时观察：

- shared memory/block；
- registers/thread；
- resident blocks/SM；
- global-load latency；
- barrier stall；
- 最终 TFLOP/s。

完成标准：在目标大 K shape 上获得稳定收益，且没有让其他主要 shape 因资源膨胀而明显退化。

## 7. P5：解决低 block 数问题

当 `ceil(M/BM) × ceil(N/BN)` 很小时，仅优化单个 block 的效率没有用。可按复杂度递进：

1. 先使用更小的 BM/BN 增加 grid；
2. 对 K 很大时实现 split-K；
3. 处理 partial C 的归约成本和确定性；
4. 若目标 workload 值得，再研究 Stream-K 或 persistent scheduling。

split-K 不是免费优化：它会增加临时空间、归约、写流量，也可能改变数值顺序。dispatcher 必须只在额外并行度收益大于归约成本时启用。

## 8. P6：增加 Tensor Core 路径

若目标是现代训练或推理 workload，只优化 FP32 SIMT 不够。建议按以下顺序：

- TF32 输入语义 + FP32 accumulation；
- FP16 + FP32 accumulation；
- BF16 + FP32 accumulation；
- 必要时融合 bias/activation。

每种模式都要明确：

- 输入和 accumulator 类型；
- 是否允许 reduced precision；
- 容差标准；
- Tensor Core 不可用时的 fallback；
- 对齐和 layout 限制。

可以借鉴 CUTLASS 的层次化 GEMM 结构与 profiler，但仍应保留本仓库易读、可逐步验证的学习目标。

## 9. P7：可靠性与发布门槛

建议把“生产可用”定义为可检查的门槛。

### 正确性

- 所有声明支持的 dtype/layout/shape 通过参考实现对比；
- 边界、随机、NaN/Inf 测试齐全；
- memcheck/racecheck 无错误；
- 明确误差与确定性契约。

### 性能

- 目标 workload 矩阵有可复现基线；
- 自动记录对 cuBLAS 的比值，而不是只保留最好结果；
- 为关键 shape 设置合理回归阈值；
- 无严重 shape 性能悬崖；
- dispatcher 开销可忽略或已计入。

### 集成

- 多 GPU device/stream 正确；
- CUDA Graph capture 可用；
- 支持的 SM 架构和 CUDA 版本有明确矩阵；
- 构建、安装、ABI 和错误信息可维护；
- 不支持的情况可靠回退。

## 10. 推荐执行顺序

近期最值得做的五件事：

1. 修复 `beta=0` 仍读取 C 和输出清零问题。
2. 增加覆盖方阵、瘦长、小 K、大 K、边界维度的 shape sweep，并生成 `custom/cuBLAS` 热力图。
3. 为现有快 kernel 增加 tail-safe 路径和对齐 fallback。
4. 实现至少 2–3 组 tile，建立第一个基于 shape 的 dispatcher。
5. 在干净基线上实现 `cp.async` pipeline，再决定 split-K 和 Tensor Core 的优先级。

建议将“达到 cuBLAS 95%”改成带适用域的表述，例如：

> 在 RTX 3060 Laptop、FP32 且禁用 TF32、M/N 为 128 的倍数、指定 shape 集合与纯 kernel 计时条件下，最佳自定义 kernel 达到对应 `torch.matmul` 性能的 95% 以上。

这种表述既保留了当前成果，也不会把局部结论误写成生产级保证。

## 11. 参考资料

- [cuBLAS Documentation](https://docs.nvidia.com/cuda/cublas/)
- [CUTLASS Efficient GEMM](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/efficient_gemm.html)
- [CUTLASS Profiler](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/profiler.html)
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [Compute Sanitizer Documentation](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/index.html)
- [CUDA Graphs](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#cuda-graphs)
