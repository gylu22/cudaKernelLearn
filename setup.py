import os
from pathlib import Path

from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# Keep the baseline build conservative. Profiling experiments should change one
# flag at a time so their effects remain attributable.
DEBUG = False

ROOT = Path(__file__).parent.resolve()
INCLUDE_DIR = ROOT / "csrc" / "include"

common_compile_args = {
    "cxx": ["-O3", "-std=c++17"],
    "nvcc": [
        "-O3",
        "-std=c++17",
        "-lineinfo",
        "--ptxas-options=-v",
    ],
}
if DEBUG:
    common_compile_args["cxx"] = ["-O0", "-g", "-std=c++17"]
    common_compile_args["nvcc"] = ["-O0", "-g", "-G", "-std=c++17"]

sgemm_compile_args = {
    key: list(flags) for key, flags in common_compile_args.items()
}
if os.environ.get("CUDA_KERNEL_LAB_SGEMM_FAST_MATH") == "1":
    sgemm_compile_args["nvcc"].append("--use_fast_math")

setup(
    name="cuda-kernel-lab",
    version="0.2.0",
    packages=find_packages(include=("cuda_kernel_lab", "cuda_kernel_lab.*")),
    ext_modules=[
        CUDAExtension(
            name="flash_attn_cuda",
            sources=[
                "csrc/bindings/flash_attn_api.cpp",
                "csrc/kernels/simt/fwd_fp32_00_pairwise_reg_scores.cu",
                "csrc/kernels/simt/fwd_fp32_01_pairwise_smem_scores.cu",
                "csrc/kernels/simt/fwd_fp32_02_pairwise_compact_smem_scores.cu",
                "csrc/kernels/tensor_core/fwd_tf32_00_wmma.cu",
            ],
            include_dirs=[str(INCLUDE_DIR)],
            extra_compile_args=common_compile_args,
        ),
        CUDAExtension(
            name="sgemm_cuda",
            sources=[
                "csrc/bindings/sgemm_api.cpp",
                "csrc/sgemm/kernels/simt/sgemm_fp32.cu",
            ],
            include_dirs=[str(INCLUDE_DIR)],
            extra_compile_args=sgemm_compile_args,
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
