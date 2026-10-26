# 多级 Pipeline：用在途数据隐藏延迟

## 1. 两级 pipeline 为什么可能不够

两级 ring buffer 在计算 stage `k` 时加载 `k+1`。它能完全隐藏加载延迟的前提近似为：

```text
当前 stage 的可重叠计算时间 >= 下一 stage 的有效加载延迟
```

若 Tensor Core 很快、K tile 很小或内存延迟较高，计算结束时下一 stage 仍未 ready。
这不是“双缓冲失效”，而是 overlap 窗口不足。

## 2. N-stage 的结构

N-stage pipeline 通常包含：

1. **Prologue**：预先提交若干 stage；
2. **Steady state**：等待最老 stage、消费它，同时提交未来 stage；
3. **Epilogue**：停止生产后排空剩余 stage。

ring index：

```cpp
read_stage = tile % Stages;
write_stage = (tile + prefetched) % Stages;
```

真正困难的是 ownership 和同步：

- 写入前确认旧消费者已经 release；
- 读取前确认对应 producer 已 commit 且 copy 已完成；
- block 内协作 copy 要保证消费者能看到其他线程写入的数据；
- tail stage 必须 zero-fill、predicate 或走安全路径；
- prologue/epilogue 的提交次数必须与 wait/release 匹配。

## 3. Stage 数量不是越多越好

每增加一个 stage 通常增加：

- `tile_A + tile_B` 的 shared memory；
- pipeline state/barrier；
- 地址和索引寄存器；
- prologue/epilogue 开销；
- 同时驻留 block 下降的风险。

Stage 太多还可能超过异步 group 或 barrier 的架构限制。最佳值由 tile、K 循环、
计算/加载比例和目标 GPU 决定，常见数值不能当作规则。

## 4. 逐步实现

1. 单 buffer 同步 copy，建立正确性和指标基线。
2. 两 buffer，但仍同步加载；验证 ring index 和 barrier。
3. 换成异步 copy，检查生成指令。
4. 扫描 2/3/4/... stages，每个版本记录资源和 duration。
5. 若增加 stage 不再降低等待，停止增加。

## 5. NCU 证据

判断 pipeline 有效应同时看到：

- duration 下降；
- memory dependency stall 或 wait 减少；
- scheduler issue efficiency 提高；
- SMEM 增长没有造成更严重的 residency/occupancy 限制；
- copy 与 compute 路径都得到合理利用；
- 结果仍通过相同 correctness suite。

不要宣称“100% 覆盖延迟”；即使 steady state 完全重叠，prologue、epilogue、tail、
barrier 和资源冲突仍存在。

## 6. Ampere 与 Hopper

- Ampere 常用 LDGSTS/`cp.async` 管理小粒度 global→shared copy；
- Hopper 提供 TMA 等 bulk/tensor-aware 数据移动，并配合更丰富的 barrier 和
  warp-specialized producer/consumer 模式。

两者的 copy 粒度、地址描述符和同步协议不同。先写清目标架构，再选择 API。

## 参考资料

- [CUDA Programming Guide: Multi-Stage Asynchronous Data Copies](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [CUDA Programming Guide: Tensor Memory Accelerator](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)

