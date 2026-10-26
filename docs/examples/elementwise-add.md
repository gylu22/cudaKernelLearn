# 示例：Elementwise Add

向量加法适合作为第一个 CUDA kernel：算法简单，可以把注意力放在索引、边界、stream、错误检查和测量上。它通常是 memory-bound。

## 1. 计算契约

```text
out[i] = a[i] + b[i],  0 <= i < n
```

输入是同一 CUDA device 上 shape、dtype 相同的连续 FP32 向量。

## 2. Kernel 与 launcher

```cpp
#include <c10/cuda/CUDAException.h>
#include <c10/util/Exception.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <limits>

__global__ void add_kernel(const float* __restrict__ a,
                           const float* __restrict__ b,
                           float* __restrict__ out,
                           int64_t n) {
  const int64_t i =
      static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) out[i] = a[i] + b[i];
}

void launch_add(const float* a, const float* b, float* out,
                int64_t n, cudaStream_t stream) {
  if (n == 0) return;
  constexpr int threads = 256;
  const int64_t blocks = (n + threads - 1) / threads;
  TORCH_CHECK(blocks <= std::numeric_limits<int>::max(),
              "input is too large");
  add_kernel<<<static_cast<int>(blocks), threads, 0, stream>>>(
      a, b, out, n);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}
```

这里使用 64 位索引，处理 `n == 0`，接收调用方 stream，并在 launch 后检查错误，但不做设备级同步。PyTorch 边界还应检查 CUDA、dtype、shape、device 和 contiguous，并使用 `CUDAGuard` 与 current stream。参见 [PyTorch CUDA 扩展](../tooling/pytorch-extensions.md)。

## 3. 正确性测试

至少覆盖：

- `n = 0, 1, 31, 32, 255, 256, 257`；
- 很大的非整除长度；
- 正数、负数、零、`inf` 和 `nan`；
- 不同 device；
- 非 contiguous 输入的拒绝或显式转换；
- 非默认 stream 下的依赖关系。

结果可与 `torch.add` 比较；包含 `nan` 时应明确 `equal_nan` 语义。

## 4. 性能模型

每个元素至少读取两个 `float`、写一个 `float`，约产生 12 byte 显式全局内存流量，只做一次加法：

```text
arithmetic intensity ≈ 1 FLOP / 12 byte
effective_bandwidth = 3 × n × sizeof(float) / elapsed_time
```

大向量的上限通常是显存带宽，小向量更可能受 kernel launch 和同步开销支配。基准测试应预热，用同一 stream 上的 CUDA events 计时，并说明是否包含分配和拷贝。

Nsight Compute 中可关注 global memory sector/efficiency、DRAM throughput、Memory SOL 和 duration。连续访问时相邻线程应访问相邻元素。`float4` 向量化只有在对齐、长度和生成代码合适时才可能获益，必须实测。

## 5. 练习

1. 支持任意 strided tensor并衡量通用索引成本；
2. 融合 bias 与 activation；
3. 实现 FP16/BF16 并检查向量化；
4. 写一个 Triton 版本进行对照；
5. 用[实验模板](../experiment-template.md)记录结果。
