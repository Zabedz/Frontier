#!/usr/bin/env bash
# run_variants.sh <variant> ... : run each variant, choosing the venv by its
# backend.inference_backend. Meant to run ON THE POD inside a run_job job.
#   hf              -> venv-A (uv sync --group hf), `frontier run`
#   vllm|llama_cpp  -> venv-B, `frontier-quantize run` (write checkpoint) then `frontier run`
# Track-B latency is off by default and enabled with TRACKB_LATENCY=1. Rows are
# append-only and resume-skip keyed, so a row banked without latency stays without it:
# run the serve-and-bench flow's one watched validation run first (docs/decisions.md
# 2026-07-17), then batch with TRACKB_LATENCY=1 so every banked row carries its latency.
set -uo pipefail
cd /workspace/frontier
CKPT=/workspace/checkpoints
TRACKB_LAT_FLAG="--skip-latency"
[ "${TRACKB_LATENCY:-0}" = "1" ] && TRACKB_LAT_FLAG=""

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
        # fp16-vllm (the fidelity gate) has no quant block and serves the base model;
        # only variants that produce a checkpoint go through frontier-quantize.
        if grep -qE '^quant:' "configs/variants/$v.yaml"; then
          frontier-quantize run --config "configs/variants/$v.yaml" --checkpoints "$CKPT" \
            || { echo "QUANT_FAILED $v"; exit 1; }
        fi
        frontier run --config "configs/variants/$v.yaml" --mode full --checkpoints "$CKPT" \
          $TRACKB_LAT_FLAG ) || echo "VARIANT_FAILED $v"
      ;;
    *) echo "UNKNOWN_BACKEND $v: $b" ;;
  esac
done
echo ALL_VARIANTS_DONE
