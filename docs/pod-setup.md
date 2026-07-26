# Pod setup

Provision the pod with an r580 or newer driver, 60GB of container disk, and TCP port 22
exposed. The driver is the hard one: venv-B follows vLLM onto CUDA 13
(`docs/decisions.md` 2026-07-26), and an r570 driver caps at CUDA 12.8, which starts the
HF, quantise, and GGUF paths fine and fails vLLM with
`cudaErrorInsufficientDriver`. Port 22 is what lets the harness rsync; the
`ssh.runpod.io` proxy forces a PTY and corrupts rsync's stream.

The GPU stack is pod-only and splits into two separate venvs. The two tracks pin different,
non-overlapping transformers versions and cannot co-resolve in one environment: Track A's HF
backend tracks the latest transformers 5.x (the `dtype=` API), while the Track-B
serving/quantise stack caps transformers at `<=5.10.1` (llmcompressor) and vLLM fixes its own
torch build. So the pod holds two venvs, and the batch driver picks one per variant by
`backend.inference_backend`. Nothing here is on the laptop: venv-A's heavy deps live in the
`hf` group (linux-marked where CUDA-only), and venv-B is installed straight on the pod, not
as a locked group, so `uv lock`/`uv sync` on a laptop stays CPU-clean.

## venv-A: Track A (the HF backend)

```bash
uv sync --group hf     # on the pod: transformers 5.x, torch (CUDA), accelerate, bitsandbytes
```

Runs the FP16 baseline and the bitsandbytes variants (`nf4`, `int8`) through the HF
backend and the WP4 CUDA-event latency rig. `bitsandbytes` is in the `hf` group behind a
`sys_platform == 'linux'` marker, so it installs on the pod and is skipped on the laptop.

Variants on venv-A (`backend.inference_backend: hf`): `fp16`, `int4-nf4`,
`int8-weightonly`.

## venv-B: Track B (serving and quantise)

A second, separate venv on the local disk (`/root`, not `/workspace`, so torch imports in
~4s instead of ~48s off MooseFS), pinned against the real CUDA build on the pod rather than
in `uv.lock`. The install is one uv resolution pass: a split install lets vLLM's unbounded
`transformers>=5.5.3` pull a newer transformers, which drops llmcompressor to its
transformers-4 line and fails with `Could not import module 'PreTrainedModel'`. The
`--override` reconciles vLLM's `compressed-tensors==0.17.0` with llmcompressor's `==0.17.1`.
The torch trio is pinned to its `+cu128` builds by name; `UV_TORCH_BACKEND=cu128` cannot be
used because it routes every torch-family package to the cu128 index exclusively, and vLLM's
`torchcodec>=0.14` only exists on PyPI (the cu128 index stops at 0.11.1). torchcodec 0.14.0
is the torch 2.11 pairing (its metadata declares no torch requirement, so the resolver
cannot check this), and `--index-strategy unsafe-best-match` lets each pin pull from
whichever index has that exact build.

```bash
uv venv /root/.venv-trackB && source /root/.venv-trackB/bin/activate
printf 'compressed-tensors==0.17.1\n' > /tmp/ct-override.txt
uv pip install --override /tmp/ct-override.txt \
  --extra-index-url https://download.pytorch.org/whl/cu128 \
  --index-strategy unsafe-best-match \
  -r pyproject.toml \
  torch==2.11.0+cu128 torchaudio==2.11.0+cu128 torchvision==0.26.0+cu128 \
  torchcodec==0.14.0 \
  vllm==0.25.1 transformers==5.10.1 compressed-tensors==0.17.1 \
  llmcompressor==0.12.0 accelerate==1.13.0 gptqmodel==7.1.0 torchao==0.17.0
CMAKE_ARGS="-DGGML_CUDA=on" uv pip install llama-cpp-python --no-cache-dir
uv pip install -e . --no-deps              # frontier + frontier-quantize entry points;
                                           # base deps rode the single pass above
```

`scripts/pod/bootstrap_trackb.sh` runs exactly this, and the pre-baked image
(`docker/`) bakes it in so the pod skips the install. Verify `torch.version.cuda` matches
the pod after install. Do not install `bitsandbytes` here (it is Track-A-only), and do not
loosen the transformers pin above 5.10.1 (llmcompressor rejects it). Runs `frontier-quantize`
(the compressed-tensors and GGUF producers) and the `vllm` / `llama_cpp` `frontier run`.

The image also bakes the llama.cpp native tools at `/opt/llama.cpp`, pinned to the commit
llama-cpp-python 0.3.34 vendors so the bench binary and the eval bindings run the same
build. The GGUF producer reads `FRONTIER_LLAMA_CPP_REPO` (the repo, for
`convert_hf_to_gguf.py`; its `gguf-py` is installed into venv-B with `--no-deps`) and
`FRONTIER_LLAMA_QUANTIZE_BIN`; the latency probe calls `llama-bench` off `PATH`
(`/opt/llama.cpp/build/bin`). `bootstrap_trackb.sh` writes all three into the job env and
warns when the tools are missing (a pod not on the pre-baked image): GGUF variants then
cannot run, the vLLM path is unaffected.

Variants on venv-B: `int4-gptq`, `int4-awq`, `int8-w8a8` (`vllm`), `gguf-q4_k_m`,
`gguf-q5_k_m` (`llama_cpp`), and the `fp16-vllm` fidelity gate.

## The batch driver picks the venv per variant

The driver reads `backend.inference_backend`: `hf` runs `frontier run` in venv-A;
`vllm` and `llama_cpp` run `frontier-quantize` (to write the checkpoint) then `frontier
run` (to serve it) in venv-B. The two venvs never run in one process, which is the whole
point of the split: the transformers pins are irreconcilable, so keeping them apart is a
correctness requirement, not a convenience.

The vLLM logit provider (`frontier.backends.vllm`) requests `logprobs=-1` (full vocab)
and reads only the known answer-letter ids out of each returned dict, so every candidate
is present at any rank; under V1 the returned logprobs are the raw model output, before
any logits processor and before temperature, which is what makes the candidate softmax
reproduce the HF baseline. `logprobs=-1` needs `max_logprobs=-1` at `LLM(...)`
construction, or the default cap of 20 rejects the request. Confirm the installed build
accepts `max_logprobs=-1`; if it caps it, pass the tokenizer vocab size instead. If the
per-position full-vocab dict build turns out to dominate wall-clock on the pod, the
documented fallback is `prompt_logprobs` with an appended candidate letter (both are
correct under V1; this is a cost choice, not a correctness one).

The llama.cpp provider loads with `logits_all=True` so `scores` is sized `(n_ctx, vocab)`
and the last-position read is unambiguous. That buffer costs `n_ctx * vocab * 4` bytes of
host RAM (~1.25GB at `n_ctx=2048`, ~2.5GB at 4096), allocated once and reused across
prompts via `reset()`, comfortable on the 50GB-RAM pod. Size `n_ctx` to the longest
chat-wrapped MMLU/ARC prompt with headroom (2048 is ample for a zero-shot single-token
MCQ). Full offload is `-ngl 99` (`gpu_offload_layers: 99`); the llama.cpp latency probe
asserts the reported offloaded-layer count equals the model's layer count before it
trusts any decode number, so a partial offload fails loudly instead of reporting a
CPU-bound tok/s.

The Track-B vLLM latency probe is not yet correct: see the `NativeVllmLatency` gate in
`docs/decisions.md` (2026-07-16). It must be fixed against real vLLM on the pod before any
vLLM latency number is recorded.

Unsloth (optional QAT accelerator, GPU-only) is installed into venv-B per its current
install instructions; it is never on the CPU smoke path.

## Disk layout: local-disk venv, persistent volume for weights

The pod's venvs live on the local container disk. Ask for 60GB of it: the pre-baked
image's own layers do not count against that space, but a cold install does, and venv-A
alone measures 7.6GB with venv-B larger still. Everything large lives on the persistent
volume, set before the first install:

- `export HF_HOME=/workspace/hf` so the base Qwen2.5-3B (~6GB) and every downloaded
  model land on the volume, not the local disk.
- `export UV_CACHE_DIR=/workspace/uv-cache` so vLLM's large wheels do not fill local
  disk during install.
- Run `frontier run` and `frontier-quantize` with `--checkpoints /workspace/checkpoints`
  so quantised checkpoints go on the volume. Budget: four compressed-tensors dirs at
  ~2-3GB, an f16 GGUF intermediate at ~6GB, and each k-quant at ~2GB, well past 12GB.

`checkpoint_path` (WP5, `frontier.quantize.paths`) derives every path from the config, so
the volume is a single flag rather than scattered path logic.

## Track B calibration gate

Before any vLLM ECE number sits next to an HF ECE number, score an unquantised FP16
Qwen2.5-3B through both the HF backend and the vLLM backend on the same MMLU slice and
assert the candidate-softmax and ECE agree (WP5, `tests/backends/test_vllm_fidelity.py`,
driven by `configs/variants/fp16-vllm.yaml`). This is the architecture rule that a Track
B method reproduces the Track A baseline before it enters the calibration analysis.

## Bring-up checklist

1. Log the pod. Every change to the pod's state goes in `SSH_CHANGELOG.md`
   (git-ignored, local), newest first, with a revert command. Keep work in the
   container / a scratch dir, off shared disks.
2. Record the environment for provenance: `nvidia-smi` (GPU model, driver, CUDA),
   card memory, and the pod id. These populate the result-row `Provenance` and
   `Backend` fields.
3. Probe clock control (it usually fails on RunPod; find out on the real pod):
   ```bash
   nvidia-smi -pm 1
   nvidia-smi --lock-gpu-clocks=<MHz> --lock-memory-clocks=<MHz>
   ```
   If it returns "Insufficient Permissions", set `clocks_locked=false` and fall
   back to the quiet-machine protocol in `docs/methodology.md` (log clocks and
   temperature per measurement, warm up, many repeats, report the clock range).
4. First real spend is WP3's latency rig, which measures the pod's true throughput.
   Re-confirm the WP6/WP7 training-token budgets against that number before running
   the expensive arms. Measure, do not assume.
