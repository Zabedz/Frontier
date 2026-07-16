# Pre-baked pod image: both venvs and every heavy dependency, so a fresh RunPod pod
# is ready in ~1-2 min (rsync the code, link it) instead of ~25 min of installs.
#
# The two tracks pin different transformers: Track A tracks the latest transformers 5.x
# (via uv.lock), while Track B's llmcompressor caps it at <= 5.10.1 and vllm fixes its own
# torch, so they cannot co-resolve. The image holds two separate venvs on the fast local
# disk, exactly where the pod scripts expect them:
#   /root/frontier-venv   venv-A: base + dev + hf + oracles (latest transformers 5.x, bnb)
#   /root/.venv-trackB     venv-B: vLLM + llm-compressor + llama.cpp (transformers 5.10.1)
# Neither venv installs the frontier package: the code is rsynced at pod start and
# linked with `uv pip install -e . --no-deps` (seconds), so the image never goes stale.
#
# Build (on a linux/amd64 machine or via buildx, NOT natively on an arm64 Mac):
#   docker buildx build --platform linux/amd64 -f docker/Dockerfile \
#     -t <dockerhub-user>/frontier-pod:cu128 --push .
# Then set the RunPod template's Container Image to that tag. See docker/README.md.

FROM nvidia/cuda:12.8.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.11 python3.11-venv python3.11-dev \
      git tmux rsync curl ca-certificates build-essential cmake ninja-build openssh-client \
 && rm -rf /var/lib/apt/lists/*

# uv, on PATH for every later step and at runtime
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:${PATH}"
ENV UV_PYTHON=python3.11

WORKDIR /build

# venv-A: the CPU + HF stack, deps only (the frontier package is linked at pod start).
# uv.lock is universal, so the linux resolution of the hf/oracles groups is baked here.
COPY pyproject.toml uv.lock ./
ENV UV_PROJECT_ENVIRONMENT=/root/frontier-venv
RUN uv sync --frozen --group hf --group oracles --no-install-project

# venv-B: the Track-B serving/quantise stack in its own venv. This MUST be one uv
# resolution pass. vllm 0.25.1 pins transformers>=5.5.3 with no upper bound, so a second,
# separate install re-resolves and drifts transformers off llmcompressor's window
# (<=5.10.1), dropping llmcompressor to its transformers-4 line whose compressed-tensors
# does `from transformers import PreTrainedModel` against the 4.x lazy-module layout and
# fails with "Could not import module 'PreTrainedModel'". The --override reconciles the one
# real clash: vllm pins compressed-tensors==0.17.0, llmcompressor pins ==0.17.1 (an
# API-identical patch bump). UV_TORCH_BACKEND=cu128 makes torch 2.11.0 resolve to the CUDA
# 12.8 wheel; the version pin alone does not choose the CUDA build. The import probe below
# is the exact regression guard (it triggers the PreTrainedModel path) and runs CPU-only.
# The GGUF build is non-fatal (llama.cpp is a latency track only).
RUN uv venv /root/.venv-trackB \
 && . /root/.venv-trackB/bin/activate \
 && printf 'compressed-tensors==0.17.1\n' > /tmp/ct-override.txt \
 && UV_TORCH_BACKEND=cu128 uv pip install --override /tmp/ct-override.txt \
      vllm==0.25.1 transformers==5.10.1 compressed-tensors==0.17.1 \
      llmcompressor==0.12.0 accelerate==1.13.0 gptqmodel==7.1.0 torchao==0.17.0 \
 && python -c "import transformers, compressed_tensors, llmcompressor; print('trackB import OK: transformers', transformers.__version__, 'ct', compressed_tensors.__version__)" \
 && ( CMAKE_ARGS="-DGGML_CUDA=on" uv pip install llama-cpp-python --no-cache-dir \
      || echo "WARN: llama-cpp-python CUDA build failed at image time; GGUF unavailable" )

# Weights and checkpoints go on the pod's persistent volume, not the image.
ENV HF_HOME=/workspace/frontier/hf-cache

CMD ["/bin/bash"]
