#!/usr/bin/env bash
# Track-B pod setup: a SECOND venv, separate from the hf venv, because llmcompressor caps
# transformers at <=5.10.1 while venv-A tracks the latest transformers 5.x, and vllm fixes
# its own torch, so the two cannot co-resolve. Run this after bootstrap.sh (venv-A) on the
# pod. On the pre-baked image venv-B already exists, so this links the code and returns.
# Weights and the uv cache go on the persistent volume; the venv itself stays on the
# fast local disk. Launches the install as a detached tmux job named `setup-trackb`.
set -euo pipefail
_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "$_HERE/lib.sh"

echo "[trackb] writing venv-B env"
pod_ssh "cat > $POD_DIR/.jobs/env-trackb" <<'ENV'
export PATH=$HOME/.local/bin:$PATH
export VIRTUAL_ENV=/root/.venv-trackB
export PATH=/root/.venv-trackB/bin:$PATH
export HF_HOME=/workspace/frontier/hf-cache
export UV_CACHE_DIR=/workspace/uv-cache
ENV

echo "[trackb] launching venv-B setup as job 'setup-trackb' (pre-baked image links in seconds; a cold pod builds it in one uv pass)"
# One resolution pass with a compressed-tensors override; docker/Dockerfile explains why a
# split install reproduces the PreTrainedModel import failure. Do NOT add bitsandbytes here.
"$_HERE/run_job.sh" setup-trackb '
set -e
export PATH=$HOME/.local/bin:$PATH
export HF_HOME=/workspace/frontier/hf-cache UV_CACHE_DIR=/workspace/uv-cache
mkdir -p /workspace/uv-cache /workspace/checkpoints
if [ -d /root/.venv-trackB ] && /root/.venv-trackB/bin/python -c "import transformers, compressed_tensors, llmcompressor" 2>/dev/null; then
  echo "venv-B is pre-baked and imports cleanly; skipping the reinstall"
  source /root/.venv-trackB/bin/activate
else
  echo "venv-B absent or broken; building it in one uv pass"
  uv venv --clear /root/.venv-trackB
  source /root/.venv-trackB/bin/activate
  printf "compressed-tensors==0.17.1\n" > /tmp/ct-override.txt
  UV_TORCH_BACKEND=cu128 uv pip install --override /tmp/ct-override.txt \
    vllm==0.25.1 transformers==5.10.1 compressed-tensors==0.17.1 \
    llmcompressor==0.12.0 accelerate==1.13.0 gptqmodel==7.1.0 torchao==0.17.0
fi
cd /workspace/frontier && uv pip install -e .
python -c "import vllm, llmcompressor, transformers, torch; print(\"vllm\", vllm.__version__, \"transformers\", transformers.__version__, \"torch\", torch.__version__, \"cuda\", torch.cuda.is_available())"
# GGUF is a latency track only; a llama.cpp CUDA build failure must not sink the vLLM path
if ! /root/.venv-trackB/bin/python -c "import llama_cpp" 2>/dev/null; then
  CMAKE_ARGS="-DGGML_CUDA=on" uv pip install "llama-cpp-python" --no-cache-dir || echo "WARN: llama-cpp-python CUDA build failed; GGUF unavailable, vLLM path intact"
fi
'
echo "[trackb] launched. Watch: WATCH_MAX_SECONDS=1800 scripts/pod/watch.sh setup-trackb"
