#!/usr/bin/env bash
set -euo pipefail

python_bin="${PYTHON_BIN:-python}"
operation="${1:-flash}"

usage() {
  echo "Usage:"
  echo "  ./profile.sh flash [backend] [seq_len]"
  echo "  ./profile.sh sgemm [backend] [m] [n] [k]"
  echo
  echo "PYTHON_BIN may select the active PyTorch interpreter."
}

if [[ "${operation}" == "-h" || "${operation}" == "--help" ]]; then
  usage
  exit 0
fi

if ! command -v ncu >/dev/null 2>&1; then
  echo "error: ncu was not found in PATH" >&2
  exit 127
fi

mkdir -p artifacts/ncu

case "${operation}" in
  flash|flash-attention)
    implementation="${2:-simt_fp32_pairwise_smem_scores}"
    seq_len="${3:-1024}"
    output="artifacts/ncu/flash_${implementation}_n${seq_len}"
    ncu \
      --set full \
      --force-overwrite \
      --nvtx \
      --nvtx-include "${implementation}/" \
      --launch-count 1 \
      -o "${output}" \
      "${python_bin}" benchmarks/profile_one.py \
        --implementation "${implementation}" \
        --batch 2 \
        --heads 8 \
        --seq-len "${seq_len}" \
        --warmup 10
    ;;
  sgemm)
    implementation="${2:-simt_fp32_01_smem_tiled}"
    m="${3:-4096}"
    n="${4:-4096}"
    k="${5:-4096}"
    output="artifacts/ncu/sgemm_${implementation}_m${m}_n${n}_k${k}"
    ncu \
      --set full \
      --force-overwrite \
      --nvtx \
      --nvtx-include "${implementation}/" \
      --launch-count 1 \
      -o "${output}" \
      "${python_bin}" benchmarks/profile_sgemm.py \
        --implementation "${implementation}" \
        --m "${m}" \
        --n "${n}" \
        --k "${k}" \
        --warmup 10
    ;;
  *)
    # Preserve the old two-argument FlashAttention form:
    # ./profile.sh <backend> <seq_len>
    implementation="${operation}"
    seq_len="${2:-1024}"
    output="artifacts/ncu/flash_${implementation}_n${seq_len}"
    ncu \
      --set full \
      --force-overwrite \
      --nvtx \
      --nvtx-include "${implementation}/" \
      --launch-count 1 \
      -o "${output}" \
      "${python_bin}" benchmarks/profile_one.py \
        --implementation "${implementation}" \
        --batch 2 \
        --heads 8 \
        --seq-len "${seq_len}" \
        --warmup 10
    ;;
esac
