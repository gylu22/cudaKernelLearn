# Archived SGEMM experiments

这里保存合并仓库前尚未纳入统一输入契约、测试和构建的实验：WMMA、block
swizzle、cuBLAS wrapper，以及原始 JIT benchmark。

这些文件不参与默认构建。将其中一个实验提升为活动 backend 时，应先拆出
launcher，通过 `csrc/bindings/sgemm_api.cpp` 注册，并补充正确性和 stream 测试。

