
在 CUDA kernel 的性能分析（Profiling）中，缓存命中率（Cache Hit Rate）和缓存未命中（Cache Miss）是衡量访存效率的核心指标。GPU 的内存架构是分层的，通常数据请求会按照 **L1/Tex Cache -> L2 Cache -> DRAM (Global Memory)** 的顺序进行。

如果在分析工具（如 Nsight Compute）中观察到严重的 Cache Miss，通常意味着你的内存访问模式与 GPU 的硬件设计不匹配。以下是导致这三层存储结构出现 Cache Miss 的常见原因：

---

### **1. L1 / Texture Cache Miss（一级缓存 / 纹理缓存未命中）**

L1 缓存是每个 SM（流多处理器）私有的，容量非常小（通常在几十到一百多 KB 之间）。它的主要作用是服务于同一 SM 内的线程。L1 Cache Miss 通常由以下原因引起：

- **非合并的内存访问 (Uncoalesced Memory Access)：** 这是最常见的原因。在 CUDA 中，一个 Warp（32个线程）的访存请求会被硬件尝试合并为少数几个 32-byte 或 128-byte 的事务。如果这 32 个线程访问的内存地址是不连续的、跨越多个 Cache Line（例如步长访问、随机索引），就会触发大量不必要的 Cache Line 加载，导致 L1 Miss 飙升。
    
- **寄存器溢出 (Register Spilling)：** 当 kernel 中使用的局部变量过多，超出了 SM 分配给该线程的寄存器上限时，数据会被溢出到本地内存（Local Memory）。Local Memory 物理上位于 DRAM，但会经过 L1 缓存。频繁的溢出读写会迅速占满并“冲刷”掉 L1 中的有用数据（Thrashing）。
    
- **缺乏空间局部性 (针对 Texture Cache)：** 纹理缓存（Texture Cache）专门为 2D/3D 的空间局部性进行了硬件优化。如果你将普通的一维线性数组绑定为纹理内存，却进行随机或大跨度的跳跃访问，Texture Cache 就无法发挥其预取相邻像素的优势，从而产生 Miss。
    
- **工作集过大 (Capacity Miss)：** L1 缓存太小，如果一个线程块（Thread Block）在循环中需要反复读取的数据量远大于 L1 的容量，数据在被再次重用前就已经被替换出去了。
    

### **2. L2 Cache Miss（二级缓存未命中）**

L2 缓存是整个 GPU 上所有 SM 共享的，容量相对较大（在较新的架构如 Ampere 或 Hopper 上可达 40MB - 60MB）。L2 缓存主要用于跨 SM 的数据共享和缓解全局内存带宽压力。

- **缺乏时间局部性 (Lack of Temporal Locality)：** 也就是所谓的一次性“流式访问”（Streaming）。如果你的 kernel 只是把巨大的输入数组从头到尾读一遍，做个计算然后写回，中间没有任何重复利用的数据。这种情况下，L2 缓存几乎不起作用，因为读进来的数据用完一次就没用了，必然表现为 L2 Miss 并直接穿透到 DRAM。
    
- **Grid 级别的离散访问：** 即使单个 Block 内部的访存是合并的（L1 表现不错），但如果不同的 Block 访问的 Global Memory 地址跨度极大，且毫无重叠，L2 缓存的容量很快会被耗尽，导致跨 Block 之间的数据无法在 L2 中驻留。
    
- **缓存驱逐 (Cache Eviction)：** 由于 L2 是全卡共享的，如果有并发执行的其他 kernel，或者当前 kernel 中有极其密集的内存写入操作，可能会把 L2 中原本需要读取的缓存行提前覆盖掉。
    
- **使用了绕过缓存的 PTX 指令：** 编译器有时会根据代码特征，或者你显式使用了 `__ldcs` (Load Cache Streaming) 等指令，强制数据绕过 L1 或部分缓存策略，这也会改变 L2 的命中统计。
    

### **3. DRAM (Global Memory) 流量激增**

严格来说，DRAM 是内存的最底层，不存在它自己的“Miss”（如果它里面没有数据，那就是 Page Fault 走系统内存了，这在 Unified Memory 中才会发生）。当 Profiler 提示 DRAM 瓶颈或读写量极大时，通常意味着 **L2 已经完全兜不住了，所有请求都被迫打到了显存上**。

- **不可避免的冷启动未命中 (Cold / Compulsory Miss)：** 任何数据第一次被访问时，L1 和 L2 中肯定没有，必须从 DRAM 读取。如果你的算法本身就是计算密度极低（Compute Intensity）、访存极其密集的（例如简单的 Vector Addition），那么主要的时间都会花在等待 DRAM 响应上。
    
- **对齐不当的访问：** 如果访问的数据结构没有按 32 字节或 128 字节对齐，即使逻辑上是连续的，GPU 也会向 DRAM 发起多余的内存事务（Memory Transactions），导致 DRAM 带宽浪费。