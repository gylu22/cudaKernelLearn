"""Archived pre-integration benchmark; use benchmarks/benchmark_sgemm.py."""

import time
from typing import Optional
import os 
import numpy as np

# 设置架构列表，确保在 import torch 之前
os.environ['TORCH_CUDA_ARCH_LIST'] = '8.6'
import torch
from torch.utils.cpp_extension import load

torch.set_grad_enabled(False)

# 【重点修改 1】: 强制关闭 PyTorch 的 TF32 魔法。
# 这样 reference_out 就是完美的 FP32 结果，以此作为绝对的正确基准。
torch.backends.cuda.matmul.allow_tf32 = False

# Load the CUDA kernel as a python module
lib = load(
    name="sgemm_lib",
    sources=[
        "practice/sgemm/sgemm.cu",
        # "practice/sgemm/sgemm_wmma.cu",
        # "practice/sgemm/sgemm_swizzle.cu",
        # "practice/sgemm/sgemm_cublas.cu"
    ],
    extra_cuda_cflags=[
        "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
    ],
    extra_cflags=["-std=c++17"],
    verbose=True, 
)

MAX_TFLOPS = -1

# check_accuracy 保持不变，逻辑没问题
def check_accuracy(custom_out: torch.Tensor, reference_out: torch.Tensor, tag: str, atol: float = 1e-3, rtol: float = 1e-3):
    if custom_out.dtype != reference_out.dtype:
        reference_out = reference_out.to(custom_out.dtype)

    abs_diff = torch.abs(custom_out - reference_out)
    rel_diff = abs_diff / (torch.abs(reference_out) + 1e-8)
    
    max_abs_err = torch.max(abs_diff).item()
    max_rel_err = torch.max(rel_diff).item()
    
    max_err_idx = torch.argmax(abs_diff.flatten())
    max_err_pos = (max_err_idx // custom_out.shape[1], max_err_idx % custom_out.shape[1])
    
    passed = torch.all((abs_diff <= atol) | (rel_diff <= rtol)).item()
    
    status = "PASS" if passed else "FAIL"
    color_code = "\033[92m" if passed else "\033[91m" 
    reset_code = "\033[0m"
    
    return passed, max_abs_err, max_rel_err, max_err_pos, f"{color_code}{status}{reset_code}"


def torch_sgemm(a: torch.Tensor, b: torch.Tensor, out: Optional[torch.Tensor] = None):
    if out is None:
        return torch.matmul(a, b)
    return torch.matmul(a, b, out=out)

# 【重点修改 2】: run_benchmark 增加 atol 和 rtol 参数
def run_benchmark(
    perf_func: callable,
    a: torch.Tensor,
    b: torch.Tensor,
    tag: str,
    out: Optional[torch.Tensor] = None,
    stages: int = -1,
    swizzle: bool = False,
    swizzle_stride: int = 1,
    warmup: int = 5,
    iters: int = 20,
    show_all: bool = False,
    check_acc: bool = True, 
    atol: float = 1e-3, # 默认给 CUDA Core 用的严苛阈值
    rtol: float = 1e-3  # 默认给 CUDA Core 用的严苛阈值
):
    global MAX_TFLOPS

    M, K = a.size(0), a.size(1)
    N = b.size(1)

    if a.size(0) > 1024 or a.size(1) >= 1024 or b.size(1) > 1024:
        iters = 10

    if swizzle:
        swizzle_stride = int((int(N / 8) // 256) * 256)
        swizzle_stride = swizzle_stride if swizzle_stride >= 256 else 1
        swizzle = swizzle if swizzle_stride >= 256 else False
    else:
        swizzle_stride = 1

    reference_out = None
    if check_acc:
        reference_out = torch.matmul(a, b)
        if out is None or out.shape != reference_out.shape:
            out = torch.empty_like(reference_out)

    if out is not None:
        out.fill_(0)

    # Warmup
    if out is not None:
        for _ in range(warmup):
            if stages > 1: perf_func(a, b, out, stages, swizzle, swizzle_stride)
            else: perf_func(a, b, out)
    else:
        for _ in range(warmup): _ = perf_func(a, b)

    torch.cuda.synchronize()
    
    # Timing
    start = time.time()
    if out is not None:
        for _ in range(iters):
            if stages > 1: perf_func(a, b, out, stages, swizzle, swizzle_stride)
            else: perf_func(a, b, out)
    else:
        for _ in range(iters): out = perf_func(a, b)
    torch.cuda.synchronize()
    end = time.time()
    
    mean_time = (end - start) * 1000 / iters
    
    # Accuracy Check
    acc_info = ""
    if check_acc and out is not None and reference_out is not None:
        # 将传入的 atol 和 rtol 交给 check_accuracy
        passed, max_abs, max_rel, err_pos, status_str = check_accuracy(out, reference_out, tag, atol, rtol)
        acc_info = f" | Err: {max_abs:.2e} (rel {max_rel:.2e}) [{status_str}]"
        
    out_val = [round(v, 6) for v in out.flatten()[:3].detach().cpu().numpy().tolist()]
    out_val_str = ", ".join([f"{v:<10}" for v in out_val])

    TFLOPS = (2 * M * N * K) * 1e-9 / mean_time
    improve_str = ""
    if TFLOPS > MAX_TFLOPS:
        improve_str = f"(+{(TFLOPS - MAX_TFLOPS) / MAX_TFLOPS * 100:.2f}%)" if MAX_TFLOPS > 0 else "(NEW)"
        MAX_TFLOPS = TFLOPS

    print(f"{tag:>25}: [{out_val_str}], time:{mean_time:<8.4f}ms, "
          f"swizzle: {str('NOOP' if swizzle_stride == 1 else swizzle_stride):<4}, "
          f"TFLOPS: {TFLOPS:<6.2f}{improve_str}{acc_info}")

    return out, mean_time

# --- Main Execution ---
if __name__ == "__main__":
    Ms, Ns, Ks = [4096, 8192], [4096, 8192], [2048,4096]
    MAX_M, MAX_N, MAX_K = 16384, 16384, 8192
    
    torch.manual_seed(42)
    A = torch.randn((MAX_M, MAX_K), dtype=torch.float).cuda()
    B = torch.randn((MAX_K, MAX_N), dtype=torch.float).cuda()
    C = torch.randn((MAX_M, MAX_N), dtype=torch.float).cuda()
    torch.cuda.synchronize()

    MNKs = [(M, N, K) for M in Ms for N in Ns for K in Ks]
    
    for M, N, K in MNKs:
        MAX_TFLOPS = -1
        print("-" * 140)
        print(" " * 60 + f"M={M}, N={N}, K={K}")
        a = A[:M, :K].contiguous()
        b = B[:K, :N].contiguous()
        c = C[:M, :N].contiguous()
        torch.cuda.synchronize()

        # 【重点修改 3】: CUDA Cores 使用严格的 1e-3 阈值
        # run_benchmark(lib.sgemm_naive_f32, a, b, "f32(naive)", c, check_acc=True, atol=1e-3, rtol=1e-3)
        run_benchmark(lib.sgemm_sliced_k_f32, a, b, "f32(clice)", c, check_acc=True, atol=1e-3, rtol=1e-3)
        run_benchmark(lib.sgemm_t_8x8_sliced_k_f32x4, a, b, "f32x4(clice)", c, check_acc=True, atol=1e-3, rtol=1e-3)
        # run_benchmark(lib.sgemm_t_8x8_sliced_k_bcf_f32x4, a, b, "f32x4_bcf(clice)", c, check_acc=True, atol=1e-3, rtol=1e-3)
        run_benchmark(lib.runSgemmResolveBankExtraCol, a, b, "sgemmresolvebankextracol", c, check_acc=True, atol=1e-3, rtol=1e-3)
        run_benchmark(lib.sgemm_t_8x8_sliced_k_bcf_db_f32x4, a, b, "f32x4_bcf_db(clice)", c, check_acc=True, atol=1e-3, rtol=1e-3)
        
        # 【重点修改 4】: Tensor Core 使用宽松的阈值 (最大相对误差允许 5%)
        # 因为 K 达到 4096 时，TF32 的累加误差绝对值经常会跑到 1.0 以上，这是正常的。
        # run_benchmark(lib.sgemm_wmma_naive, a, b, "wmma(naive)", c, check_acc=True, atol=2.0, rtol=5e-2)
        # run_benchmark(lib.sgemm_wmma_shared_warp_tiling,a,b,"wmma_tiling",c, check_acc=True, atol=2.0, rtol=5e-2)
        # run_benchmark(lib.sgemm_t_8x8_sliced_k_swizzle_f32x4, a, b, "f32x4_bcf(swizzle)", c, check_acc=True, atol=1e-1, rtol=5e-2)
        # run_benchmark(lib.sgemm_wmma_shared_warp_tiling_db,a,b,"wmma_tiling_db",c, check_acc=True, atol=2.0, rtol=5e-2)

        # run_benchmark(lib.sgemm_cublas, a, b, "f32(cublas)", c, check_acc=True, atol=1e-3, rtol=1e-3)
        run_benchmark(torch_sgemm, a, b, "f32(torch)", check_acc=True, atol=1e-3, rtol=1e-3)
