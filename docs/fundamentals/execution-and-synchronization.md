# CUDA 同步、内存模型、Stream 与并发

CUDA correctness 不只取决于“有没有 barrier”，还取决于参与者范围、内存顺序、
执行顺序和地址是否存在 data race。本文把 thread/warp/block/device/system 各层的
同步工具放进同一个模型。

## 1. 先区分四件事

### 1.1 Execution barrier

参与者必须都到达某个点后才能继续。例如 `__syncthreads()`。

### 1.2 Memory ordering

约束一个线程的读写被其他线程观察到的顺序。例如 memory fence、带 memory order
的 atomic。

### 1.3 Visibility

某次写入在目标 scope 是否能被其他参与者看到。这与 cache、scope 和同步协议有关。

### 1.4 Atomicity

一次读改写不可被同作用域内其他冲突操作拆开观察。atomic 不自动等于 barrier，也
不自动给整个算法建立顺序。

很多错误来自把 fence 当 barrier，或把 atomic 当成完整同步协议。

## 2. CUDA 的线程 scope

从近到远常见为：

```text
thread
  < warp/subgroup
  < block
  < cluster（支持时）
  < device
  < system
```

同步距离越远，协调成本通常越高。高性能算法优先把频繁通信限制在 warp 或 block。

CUDA C++ 的原子与同步原语可以带 thread scope，例如 block、device、system。选择
scope 时应覆盖真实通信参与者，但不要无意义地使用更宽 scope。

## 3. Warp 内同步

### 3.1 Shuffle

`__shfl_*_sync(mask, value, ...)` 在 lane 寄存器间交换数据，适合 reduction、
broadcast、scan。

关键规则：

- mask 表示需要参与该操作的 lane；
- mask 中命名的非退出线程必须正确参与；
- partial warp/tail 不能无条件使用 `0xffffffff`；
- shuffle 只交换指定寄存器值，不替代其他 shared/global memory 通信的顺序。

完整 warp reduction 示例：

```cpp
unsigned mask = 0xffffffff;
float x = value;
for (int offset = 16; offset > 0; offset /= 2) {
    x += __shfl_down_sync(mask, x, offset);
}
```

partial warp 或分组 reduction 还要保证 mask、width、目标 lane 和 source lane 都与
真实参与者一致；不能只把 full mask 替换成 `__activemask()` 就假定算法正确。

### 3.2 `__syncwarp(mask)`

它让 mask 中参与 lane 协调，并为这些 lane 建立相应内存顺序。Volta+ independent
thread scheduling 后，任何依赖隐式 warp lockstep 的 shared/global 通信都应重新
审查。

```cpp
shared[lane] = x;
__syncwarp(mask);
y = shared[other_lane];
```

若参与者跨 warp，`__syncwarp` 不够。

## 4. Block 同步

### 4.1 `__syncthreads()`

它是 block barrier，并对 block 内相关 shared/global memory 访问提供同步语义。
典型 tile：

```cpp
load_global_to_shared();
__syncthreads();

compute_from_shared();
__syncthreads();

overwrite_shared_with_next_tile();
```

第二个 barrier 防止快 warp 在慢 warp 还读取当前 tile 时覆盖 shared buffer。

### 4.2 Barrier 放在分支中的危险

以下代码只有在条件对整个 block 一致时才合法：

```cpp
if (condition) {
    __syncthreads();
}
```

如果只有部分线程到达，可能死锁或产生未定义行为。边界处理通常采用：

```cpp
if (valid) {
    shared[tid] = input[index];
} else {
    shared[tid] = 0;
}
__syncthreads();
```

让所有线程到达 barrier，而不是让无效线程提前返回。

### 4.3 `__syncthreads_count/and/or`

它们在 barrier 的同时对 predicate 做 block 范围聚合，适合在正确同步的前提下判断
block 状态。不要为了少量统计引入昂贵的全 block barrier，先考虑 warp primitive。

## 5. Fence 不是 barrier

`__threadfence_block`、`__threadfence`、`__threadfence_system` 主要约束调用线程
之前的内存操作相对之后操作的可见顺序，scope 分别扩展到 block、device、system。

它们不会：

- 让其他线程停下等待；
- 确保其他线程已经执行到对应位置；
- 自动避免多个线程同时写同一地址；
- 替代原子 flag 的发布/获取协议。

典型跨 block producer/consumer 需要“写数据 → fence/release → atomic 发布标志”，
消费者需要相应 acquire/检查。若能拆成两个 kernel，kernel launch 边界通常更简单。

## 6. Data race 与 atomic

两个线程并发访问同一位置，至少一个是写且没有正确同步，就可能构成 data race。

atomic 可用于：

- counter；
- histogram；
- work queue；
- reduction 的最终合并；
- 状态发布。

性能风险：

- 多 lane 更新同一地址会序列化；
- 热点地址造成 contention；
- 跨更宽 scope 可能更昂贵；
- 浮点 atomic 的执行顺序不固定，结果可能非确定；
- atomic 只保护该原子操作，不保护周围普通读写。

常见优化：

```text
thread local reduction
  → warp reduction
  → block reduction
  → 每 block 一次 global atomic
```

## 7. 跨 block 同步

普通 kernel 内不能假定 block 的执行顺序，也不能实现“所有 block 都会同时驻留”的
自旋 barrier；未调度 block 可能永远等不到资源，造成死锁。

选择：

- kernel 边界作为全 grid 同步；
- cooperative launch + grid synchronization，并满足驻留条件；
- 支持架构上的 block cluster；
- 原子/队列协议，但必须证明不会因调度产生死锁。

大多数学习与生产 kernel 应优先选择多 kernel 分阶段，除非 profiling 证明 launch
开销值得更复杂的 persistent/cooperative 设计。

## 8. Host、Stream 与 Event

### 8.1 Stream 顺序

同一 stream 中提交的操作按 stream 顺序执行。不同 stream 在依赖和硬件资源允许时
可以重叠，但“不同 stream”不保证一定并发。

阻止 overlap 的常见原因：

- 使用 legacy default stream 的隐式语义；
- 显式 `cudaDeviceSynchronize`；
- 内存分配/释放或 pageable copy 引起同步；
- 两个 kernel 都占满目标资源；
- 数据依赖未用 event 正确表达；
- copy 与 compute 使用相同硬件 engine 或带宽资源。

### 8.2 Event

CUDA Event 可用于：

- stream 内 GPU 时间测量；
- stream 间依赖；
- 查询异步操作是否完成。

计时：

```cpp
cudaEventRecord(start, stream);
kernel<<<grid, block, 0, stream>>>(...);
cudaEventRecord(stop, stream);
cudaEventSynchronize(stop);
cudaEventElapsedTime(&ms, start, stop);
```

Event 测到的是两 event 在该 stream 时间线上的 GPU 时间，不是完整 CPU wall time。

### 8.3 PyTorch extension

扩展必须：

- 使用 tensor 所在 device 的 guard；
- 使用调用者 current CUDA stream；
- 不在每次调用末尾无条件 device synchronize；
- 保持 tensor 生命周期覆盖异步 kernel；
- 对多 GPU 输入明确检查 device 一致性。

本仓库 binding 是这些规则的具体实例。

## 9. 异步执行与错误

kernel launch 对 host 通常是异步的：

```cpp
kernel<<<...>>>(...);
cudaError_t launch_status = cudaPeekAtLastError();
```

`cudaPeekAtLastError` 可发现 launch configuration 等立即错误，但越界访问等执行期
错误通常在后续同步 API 暴露。

分层处理：

- library fast path：检查 launch error，不强制同步；
- correctness test：在明确边界同步；
- 调试：使用 `CUDA_LAUNCH_BLOCKING=1` 或 Compute Sanitizer；
- 生产：让上层框架保持异步语义并在合适边界报告错误。

## 10. Async copy 与 pipeline

Ampere 及以后部分路径支持 global-to-shared 异步 copy。高层可用
`cuda::memcpy_async`/pipeline，底层可涉及 `cp.async`。

它解决的是：

- 减少中间 register 搬运；
- 允许下一 tile 数据移动与当前 tile compute overlap；
- 用 producer/consumer stage 表达依赖。

它不自动保证提速。需要满足：

- 每 stage 计算足够长；
- 地址、大小和对齐适合；
- stage buffer 不把 occupancy 压得过低；
- commit/wait/barrier 正确；
- tail 的 zero-fill/predication 正确；
- profiler 显示 latency stall 与 duration 改善。

详见 [异步拷贝](../tooling/async-copy.md)和
[多级流水线](../tooling/multistage-pipeline.md)。

## 11. CUDA Graph

CUDA Graph 将一组操作及依赖实例化后重复 launch，可减少重复提交开销，适合结构
稳定、短 kernel 很多的 workload。

需要考虑：

- capture 期间 API 是否 graph-safe；
- 地址/shape 是否稳定；
- allocator 行为；
- stream capture mode；
- graph update 的限制；
- kernel 本身慢时，Graph 不能修复设备端瓶颈。

## 12. 正确性检查清单

写包含协作与同步的 kernel 时逐项回答：

1. 谁写数据，谁读数据？
2. 通信范围是 warp、block、cluster、device 还是 system？
3. 是否存在 barrier？
4. 是否建立了正确 memory ordering？
5. 所有命名参与者都会到达同步点吗？
6. partial warp 的 mask 是否正确？
7. buffer 何时可以被下一 stage 覆盖？
8. atomic scope 和 memory order 是否足够？
9. 不同 block 是否错误依赖执行顺序？
10. current stream 和 tensor lifetime 是否正确？

工具验证：

```bash
compute-sanitizer --tool memcheck ...
compute-sanitizer --tool racecheck ...
compute-sanitizer --tool synccheck ...
```

racecheck 主要分析 shared-memory hazard，不是完整的所有 global-memory race detector。

## 13. 面试问题

### `__syncthreads()` 与 `__threadfence()` 的区别？

前者是 block barrier，并提供 block 内相关内存同步；后者是调用线程的内存顺序约束，
不是 rendezvous，不会等待其他线程。

### Atomic 是否等于线程安全？

只保证指定 atomic 操作不可分割。围绕它的普通数据、发布顺序和更大的不变量仍需要
正确 memory order、scope 或 barrier。

### 为什么不同 stream 不一定并发？

并发取决于依赖、默认 stream 语义、硬件 engine、资源余量、内存带宽和是否存在
隐式同步。stream 只表达可并发的顺序关系，不承诺物理同时执行。

### 为什么 Volta+ 仍然以 warp 执行，却要求 `__syncwarp`？

warp 仍是指令发射分组，但 independent thread scheduling 允许发散线程更灵活地
前进/汇合。跨 lane 内存通信不能再依赖隐式 lockstep，需要显式同步与正确 mask。

## 14. 参考资料

- [CUDA Programming Guide: CUDA C++ Memory Model](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cuda-cpp-memory-model.html)
- [CUDA Programming Guide: C/C++ Language Extensions](https://docs.nvidia.com/cuda/cuda-programming-guide/05-appendices/cpp-language-extensions.html)
- [CUDA Programming Guide: Independent Thread Scheduling](https://docs.nvidia.com/cuda/cuda-programming-guide/03-advanced/advanced-kernel-programming.html)
- [Compute Sanitizer](https://docs.nvidia.com/compute-sanitizer/ComputeSanitizer/)
