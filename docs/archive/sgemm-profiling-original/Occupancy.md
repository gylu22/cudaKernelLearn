在 CUDA 编程和性能优化中，**Occupancy（占用率）** 是一个非常核心的概念。简单来说，它衡量的是你的 GPU 计算资源“有多满”。
![alt text](../../profiling/assets/occupancy.png)
为了让你更容易理解，我们可以用一个“厨师炒菜”**的比喻： 把 GPU 里的一个 **SM (Streaming Multiprocessor，流多处理器)** 想象成一位**厨师。 把一个 **Warp (包含32个线程的执行单元)** 想象成一道**菜**。 厨师在炒某道菜时，经常需要等水烧开或者等配菜送来（这就相当于 Warp 在等待读取全局内存，也就是**延迟**）。如果厨师只负责一道菜，那等待的这段时间他只能发呆。但如果炉子上同时放着很多道菜，当第一道菜需要等待时，厨师可以立刻转身去切第二道菜的葱花。 **炉子上同时放的菜越多，厨师发呆的时间就越少。**

Occupancy 就是在衡量你的“炉子”有多满。

---

### 一、 Occupancy 的严格定义与计算

在物理硬件上，每一个架构的 GPU，其单个 SM 能同时容纳的 Warp 数量是有物理上限的（比如某些架构下，一个 SM 最多容纳 64 个 Warps）。

Occupancy 的计算公式非常直接：

Occupancy=Maximum Supported Warps per SMActive Warps per SM​×100%
- **Active Warps per SM (活跃 Warp 数)：** 当前实际驻留在一个 SM 上的 Warp 数量。
- **Maximum Supported Warps per SM (最大支持 Warp 数)：** 该 GPU 架构的硬件物理极限。
**举个例子：** 如果你的显卡架构规定一个 SM 最多能跑 64 个 Warp。而你运行 kernel 时，因为各种资源的限制，一个 SM 上实际只驻留了 32 个 Warp。那么你这个 kernel 的 Occupancy 就是 50%。

---
### 二、 为什么要关注 Occupancy？

**Occupancy 的核心目的不是为了“算得更快”，而是为了“掩盖延迟” (Latency Hiding)。**
GPU 的全局内存读写非常慢（几百个时钟周期）。当一个 Warp 遇到访存指令被卡住时，调度器会进行零开销的上下文切换 (Context Switch)，瞬间把计算单元分配给另一个已经准备好的 Warp。 如果你的 Occupancy 很高，意味着 SM 上有大量“备胎” Warp。总有一个 Warp 是准备好进行计算的，这样计算单元就不会闲置，从而达到提升整体吞吐量的目的。

_(注：Occupancy 并非越高越好，通常达到 50% - 70% 就能很好地掩盖大部分延迟。盲目追求 100% 可能会适得其反，导致缓存抖动。)_

---

### 三、 是什么限制了你的 Occupancy？ (如何优化)

这是你在 `ncu` 调优时最需要关心的部分。一个 SM 上的空间很大，但为什么你的 Warp 放不满呢？通常是因为以下三种“木桶效应”（任何一个资源耗尽，就无法再分配新的 Block/Warp）：

#### 1. 寄存器消耗过多 (Registers Per Thread)

每个 SM 上的物理寄存器总数是固定的（例如 64K 个 32-bit 寄存器）。

- 如果你写了一个很复杂的 kernel，每个线程需要使用 128 个寄存器。
- 那么一个包含 256 个线程的 Block 就会吃掉大量的寄存器。
- 很快，SM 的寄存器就被分光了，哪怕 SM 上还有空余的线程槽位，也无法再容纳新的 Block 了。
- **ncu 对应指标：** 检查 Summary 里的 `Registers Per Thread`。如果太高，可以尝试在编译时使用 `-maxrregcount` 限制寄存器数量，或者简化代码逻辑。

#### 2. 共享内存消耗过多 (Shared Memory Per Block)

每个 SM 的 Shared Memory 大小也是固定的（比如 64KB 或 96KB）。

- 如果你的一个 Block 申请了 48KB 的共享内存。
- 即使一个 SM 理论上能放下 16 个 Block，但由于共享内存只够分给 1 到 2 个 Block，剩余的 Block 只能在外面排队。

#### 3. 线程块配置不合理 (Block Size / Grid Size)

硬件对 SM 上能同时驻留的 Block 数量和 Thread 数量都有硬性规定。

- 如果你设置的 Block 很小（例如每个 Block 只有 32 个线程）。
- SM 很快就会达到“最大允许驻留的 Block 数量”的上限，但此时总的线程数还远远没有达到 SM 的容量极限，这就造成了空间浪费。

---

### 四、 在 Nsight Compute (ncu) 中看什么？

在 `ncu` 的 **Details** 页面，你会看到两个与 Occupancy 相关的核心指标：

1. **Theoretical Occupancy (理论占用率)：** ncu 根据你 kernel 的寄存器使用量、共享内存使用量以及 Block 大小，计算出**理论上**你的 SM 能达到多满。如果这个值很低，说明你的**配置或者代码资源消耗**有问题。
2. **Achieved Occupancy (实际占用率)：** 在程序实际运行时，硬件监控到的真实的驻留比例。如果理论很高，但实际很低，通常说明你的 **Grid Size 太小**（总的工作量不够填满整个 GPU），或者遭遇了严重的**尾部效应**（有的 Block 早就跑完了，有的还在跑，导致后期 SM 空闲）。
