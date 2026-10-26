# Ampere 异步 global-to-shared 拷贝

本文讨论 Compute Capability 8.0 起的 LDGSTS/`cp.async` 路径。Hopper 及以后还有
TMA bulk copy，语义和使用场景不同，不应都简称为 `cp.async`。

## 1. 它解决什么

传统 C++ 写法常被编译为：

```text
global load → register → shared store
```

支持的异步路径可以直接表达 global → shared copy，减少中间寄存器需求，并允许
在 pipeline 正确安排时与计算重叠。API “异步”不等于硬件上必然完全重叠；是否
受益取决于架构、访问模式、stage 工作量和资源。

## 2. LDGSTS 的约束

按照当前 CUDA Programming Guide：

- 方向是 global memory → shared memory；
- 元素级 copy 支持 4、8、16 byte；
- 地址需满足相应自然对齐；更大的整体对齐有助于最佳性能；
- 4/8-byte copy 使用 L1 access mode，16-byte copy可以使用 L1 bypass mode；
- copy 完成与其他线程可见性是两个问题：等待异步操作后，跨线程消费通常还需要
  block/warp 协作同步。

因此“永远绕过 L1”“一定零寄存器”“永远使用 16 byte”都不是完整规则。tail、
对齐和每 lane 工作分配可能让较小粒度或 fallback 更合适。

## 3. 推荐的 C++ API

优先使用 `<cuda/pipeline>` 和 `cuda::memcpy_async`，由工具链选择可用实现。下面是
结构示意，省略了每线程地址映射和 tail：

```cpp
#include <cooperative_groups.h>
#include <cuda/pipeline>

namespace cg = cooperative_groups;

template <int Stages, int Elements>
__global__ void pipelined_kernel(const float* input, float* output) {
    __shared__ float smem[Stages][Elements];
    __shared__ cuda::pipeline_shared_state<
        cuda::thread_scope_block, Stages> state;

    auto block = cg::this_thread_block();
    auto pipe = cuda::make_pipeline(block, &state);

    // producer_acquire → memcpy_async → producer_commit
    // consumer_wait → read stage → consumer_release
    // 以 ring buffer 方式推进，另行处理 prologue/epilogue。
}
```

使用 pipeline 时所有参与线程必须遵守同一套 acquire/commit/wait/release 协议。
producer/consumer 分区和 unified pipeline 的规则不同，不能混用。

## 4. PTX 层

Ampere PTX 常见形式：

```ptx
cp.async.cg.shared.global [dst_smem], [src_global], 16;
cp.async.commit_group;
cp.async.wait_group 1;
```

内联 PTX 还需要：

- `__CUDA_ARCH__ >= 800` guard；
- 用 `__cvta_generic_to_shared`/合适 helper 转换 shared 地址；
- 正确的 32-bit shared 与 64-bit global operand constraint；
- immediate wait-group 参数；
- 对 tail 使用 `src-size`/zero-fill 语义或安全 fallback；
- 编译后检查 PTX/SASS，而不是假定 inline asm 一定无额外指令。

学习阶段建议先让同步版本和逻辑双缓冲正确，再替换 copy 原语。否则 pipeline
索引 bug、同步 bug 与 `cp.async` bug 会混在一起。

## 5. 验证是否真的 overlap

比较：

- duration；
- register、spill、dynamic SMEM 和 occupancy；
- long/short scoreboard；
- eligible/issued warps；
- memory 与 compute pipe 的并行程度；
- Source/SASS 中是否出现目标 copy 指令。

若一个 stage 的计算远短于内存延迟，两级 pipeline 可能仍会等待；增加 stage 又会
消耗 shared memory。详见[多级流水线](multistage-pipeline.md)。

## 参考资料

- [CUDA Programming Guide: Asynchronous Data Copies](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#asynchronous-data-copies)
- [libcu++ `cuda::pipeline`](https://nvidia.github.io/cccl/libcudacxx/extended_api/synchronization_primitives/pipeline.html)
- [PTX ISA: `cp.async`](https://docs.nvidia.com/cuda/parallel-thread-execution/#data-movement-and-conversion-instructions-cp-async)

