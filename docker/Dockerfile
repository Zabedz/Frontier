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
# API-identical patch bump). frontier's own base deps (-r pyproject.toml) ride the same
# pass, so venv-B is self-sufficient and the pod-start link is `uv pip install -e .
# --no-deps`: after this layer, no later resolve can move any of these pins.
#
# Torch wheels: vllm needs torchcodec>=0.14 and the cu128 index stops at 0.11.1, so
# UV_TORCH_BACKEND=cu128 (which routes every torch-family package to that index
# exclusively) cannot resolve. The torch trio is pinned to its +cu128 builds by name,
# torchcodec to 0.14.0 (the torch 2.11 pairing; torchcodec declares no torch requirement
# in its metadata, so the resolver cannot check this), and --index-strategy
# unsafe-best-match lets each pin pull from whichever index carries that exact build:
# the trio and torchao from the cu128 index, torchcodec from PyPI. torchcodec is dead
# weight here (text-only benchmark, and its decoder needs an FFmpeg the image does not
# ship); it exists to satisfy vllm's requirement.
#
# The import probe below is the exact regression guard (it triggers the PreTrainedModel
# path) and runs CPU-only; the torch assert catches a silent fallback to a non-CUDA-12.8
# build. The GGUF build is non-fatal (llama.cpp is a latency track only).
RUN uv venv /root/.venv-trackB \
 && . /root/.venv-trackB/bin/activate \
 && printf 'compressed-tensors==0.17.1\n' > /tmp/ct-override.txt \
 && uv pip install --override /tmp/ct-override.txt \
      --extra-index-url https://download.pytorch.org/whl/cu128 \
      --index-strategy unsafe-best-match \
      -r pyproject.toml \
      torch==2.11.0+cu128 torchaudio==2.11.0+cu128 torchvision==0.26.0+cu128 \
      torchcodec==0.14.0 \
      vllm==0.25.1 transformers==5.10.1 compressed-tensors==0.17.1 \
      llmcompressor==0.12.0 accelerate==1.13.0 gptqmodel==7.1.0 torchao==0.17.0 \
 && python -c "import torch, transformers, compressed_tensors, llmcompressor; assert torch.__version__ == '2.11.0+cu128', torch.__version__; print('trackB import OK: torch', torch.__version__, 'transformers', transformers.__version__, 'ct', compressed_tensors.__version__)" \
 && ( CMAKE_ARGS="-DGGML_CUDA=on" uv pip install llama-cpp-python --no-cache-dir \
      || echo "WARN: llama-cpp-python CUDA build failed at image time; GGUF unavailable" )

# llama.cpp native tools, pinned to the exact commit llama-cpp-python 0.3.34 vendors, so
# llama-bench and llama-quantize run the same llama.cpp build the eval bindings do. The
# GGUF producer runs the repo's convert_hf_to_gguf.py (its gguf-py package goes into
# venv-B; --no-deps so a second resolve can never move the venv's pins) and
# llama-quantize; the latency probe calls llama-bench off PATH. Non-fatal, matching the
# bindings build above: GGUF is a latency track and must not sink the vLLM path.
# The build container has no GPU driver, so the executables resolve libggml-cuda's
# libcuda.so.1 against the CUDA driver stub via -rpath-link: link-time resolution only,
# nothing recorded in the binary, and the pod's injected real driver is used at runtime.
RUN ( git init /opt/llama.cpp \
 && git -C /opt/llama.cpp remote add origin https://github.com/ggml-org/llama.cpp \
 && git -C /opt/llama.cpp fetch --depth 1 origin e3546c7948e3af463d0b401e6421d5a4c2faf565 \
 && git -C /opt/llama.cpp checkout FETCH_HEAD \
 && mkdir -p /tmp/cuda-stubs \
 && ln -sf /usr/local/cuda/lib64/stubs/libcuda.so /tmp/cuda-stubs/libcuda.so.1 \
 && cmake -S /opt/llama.cpp -B /opt/llama.cpp/build -G Ninja \
      -DGGML_CUDA=ON -DLLAMA_BUILD_TESTS=OFF \
      -DCMAKE_EXE_LINKER_FLAGS=-Wl,-rpath-link,/tmp/cuda-stubs \
 && cmake --build /opt/llama.cpp/build --target llama-quantize llama-bench \
 && . /root/.venv-trackB/bin/activate \
 && uv pip install --no-deps /opt/llama.cpp/gguf-py \
 && test -x /opt/llama.cpp/build/bin/llama-bench \
 && test -x /opt/llama.cpp/build/bin/llama-quantize \
 && python -c "import gguf; print('llama tools OK: gguf-py present, binaries built')" \
 || echo "WARN: llama.cpp tools build failed at image time; GGUF producer and bench unavailable" )
ENV FRONTIER_LLAMA_CPP_REPO=/opt/llama.cpp
ENV FRONTIER_LLAMA_QUANTIZE_BIN=/opt/llama.cpp/build/bin/llama-quantize
ENV PATH="/opt/llama.cpp/build/bin:${PATH}"

# Two libraries the default loader path gets wrong, both confirmed on a pod:
# nccl, because the base image's /usr/lib copy is 2.25.1 and predates ncclCommShrink,
# which llama_cpp drags in ahead of the venv's 2.28.9 and breaks a later torch import;
# and the CUDA 13 runtime, which every published vLLM wheel's extension links against and
# which ships under nvidia/cu13 rather than on the path. The latter also needs an r580+
# driver: on r570 the import resolves and CUDA init returns cudaErrorInsufficientDriver.
ENV LD_LIBRARY_PATH="/root/.venv-trackB/lib/python3.11/site-packages/nvidia/nccl/lib:/root/.venv-trackB/lib/python3.11/site-packages/nvidia/cu13/lib:${LD_LIBRARY_PATH}"

# Weights and checkpoints go on the pod's persistent volume, not the image.
ENV HF_HOME=/workspace/frontier/hf-cache

# A long-running process, so the container stays up for the pod's lifetime. An
# interactive shell here reads EOF and exits the moment the container starts without a
# TTY, which leaves the pod green in the console but its container dead and unreachable.
CMD ["sleep", "infinity"]
