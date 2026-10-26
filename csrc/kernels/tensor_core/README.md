# Tensor Core learning track

## Active baseline

`fwd_tf32_00_wmma.cu` is a fixed-D educational implementation:

- contiguous FP32 BHND input and output;
- fixed `head_dim=64`;
- Ampere SM 8.0 or newer;
- explicit FP32-to-TF32 conversion;
- WMMA `m16n16k8` for QK and PV;
- FP32 online softmax on CUDA Cores;
- 16 Query rows and 64 K/V rows per tile;
- four warps per block.

SM 8.6 compile baseline:

```text
registers/thread: 166
spill loads:       0
spill stores:      0
dynamic SMEM:      45,248 bytes
SASS:              HMMA.1684.F32.TF32
```

Profile with:

```bash
./profile.sh tensor_core_tf32_wmma 1024
```

## Suggested progression

1. Profile HMMA utilization, barriers, SMEM conflicts, and Tensor Core stalls.
2. Replace scalar row softmax with warp/subwarp reductions.
3. Reuse or compact shared-memory buffers.
4. Add logical double buffering, then `cp.async`.
5. Add FP16/BF16 storage with FP32 accumulation.
6. Move to CUTLASS/CUTE or explicit `mma.sync` after the layout is understood.

Each stage should remain selectable as an independent backend and use a
precision-appropriate correctness tolerance.
