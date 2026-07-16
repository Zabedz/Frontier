# Pod setup

The GPU stack is pod-only and splits into two separate venvs. The two tracks pin
incompatible transformers and cannot co-resolve in one environment: Track A's HF backend
uses the transformers 5.x `dtype=` API, while the Track-B serving/quantise stack caps
transformers below 5 (llmcompressor) and vLLM pulls that lower version. So the pod holds
two venvs, and the batch driver picks one per variant by `backend.inference_backend`.
Nothing here is on the laptop: venv-A's heavy deps live in the `hf` group (linux-marked
where CUDA-only), and venv-B is installed straight on the pod, not as a locked group, so
`uv lock`/`uv sync` on a laptop stays CPU-clean.

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

A second, separate venv, pinned against the real CUDA build on the pod rather than in
`uv.lock`. Install order has one coupling that bites: vLLM pins a specific torch build, so
it goes first and fixes torch and transformers; do not pre-pin torch or add transformers
5.x, which llmcompressor rejects.

```bash
uv venv /workspace/.venv-trackB && source /workspace/.venv-trackB/bin/activate
uv pip install "vllm>=0.8"                 # V1 engine; fixes torch + transformers (< 5)
uv pip install llmcompressor compressed-tensors accelerate gptqmodel torchao
CMAKE_ARGS="-DGGML_CUDA=on" uv pip install llama-cpp-python --no-cache-dir
uv pip install -e .                        # frontier + frontier-quantize entry points
```

Verify `torch.version.cuda` matches the pod after the vLLM install. Do not install
`bitsandbytes` here (it is Track-A-only), and do not install `transformers>=5` (vLLM and
llmcompressor choose the compatible `< 5` version). Runs `frontier-quantize` (the
compressed-tensors and GGUF producers) and the `vllm` / `llama_cpp` `frontier run`.

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

The pod's venv lives on the small local disk (~12GB), and the venv plus vLLM alone
approaches that cap. Everything large lives on the persistent volume, set before the
first install:

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
