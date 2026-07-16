#!/usr/bin/env bash
# run_variants.sh <variant> ... : run each variant, choosing the venv by its
# backend.inference_backend. Meant to run ON THE POD inside a run_job job.
#   hf              -> venv-A (uv sync --group hf), `frontier run`
#   vllm|llama_cpp  -> venv-B, `frontier-quantize run` (write checkpoint) then `frontier run`
# Track-B latency is skipped here: the NativeVllmLatency gate is open (docs/decisions.md),
# and ECE/accuracy are the Track-B contribution, trustworthy once the FP16-vLLM fidelity
# gate passes. Real Track-B latency is a separate, validated pass.
set -uo pipefail
cd /workspace/frontier
CKPT=/workspace/checkpoints

backend_of() { grep -E '^[[:space:]]*inference_backend:' "configs/variants/$1.yaml" | awk '{print $2}'; }

for v in "$@"; do
  b=$(backend_of "$v")
  echo "=== $v (backend=$b) $(date -u +%T) ==="
  case "$b" in
    hf)
      ( source /workspace/frontier/.jobs/env
        uv run --group hf frontier run --config "configs/variants/$v.yaml" --mode full \
          --checkpoints "$CKPT" ) || echo "VARIANT_FAILED $v"
      ;;
    vllm | llama_cpp)
      ( source /workspace/frontier/.jobs/env-trackb
        source /root/.venv-trackB/bin/activate
        frontier-quantize run --config "configs/variants/$v.yaml" --checkpoints "$CKPT" \
          || { echo "QUANT_FAILED $v"; exit 1; }
        frontier run --config "configs/variants/$v.yaml" --mode full --checkpoints "$CKPT" \
          --skip-latency ) || echo "VARIANT_FAILED $v"
      ;;
    *) echo "UNKNOWN_BACKEND $v: $b" ;;
  esac
done
echo ALL_VARIANTS_DONE
