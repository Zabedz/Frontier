#!/usr/bin/env bash
# Track-B pod setup: a SECOND venv, separate from the hf venv, because vLLM +
# llmcompressor pin transformers < 5 while the HF backend needs 5.x (they cannot
# co-resolve). Run this after bootstrap.sh (which sets up venv-A) on the pod.
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

echo "[trackb] launching venv-B install (vLLM first, then producers, then llama.cpp) as job 'setup-trackb'"
# vLLM first fixes torch + transformers (<5); do NOT add bitsandbytes or transformers>=5.
"$_HERE/run_job.sh" setup-trackb '
set -e
export PATH=$HOME/.local/bin:$PATH
export HF_HOME=/workspace/frontier/hf-cache UV_CACHE_DIR=/workspace/uv-cache
mkdir -p /workspace/uv-cache /workspace/checkpoints
uv venv /root/.venv-trackB
source /root/.venv-trackB/bin/activate
uv pip install "vllm>=0.8"
uv pip install llmcompressor compressed-tensors accelerate gptqmodel torchao
CMAKE_ARGS="-DGGML_CUDA=on" uv pip install "llama-cpp-python" --no-cache-dir
cd /workspace/frontier && uv pip install -e .
python -c "import vllm, llmcompressor, transformers, torch; print(\"vllm\", vllm.__version__, \"transformers\", transformers.__version__, \"torch\", torch.__version__, \"cuda\", torch.cuda.is_available())"
'
echo "[trackb] launched. Watch: WATCH_MAX_SECONDS=1800 scripts/pod/watch.sh setup-trackb"
