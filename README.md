# Frontier

A benchmark for the accuracy, latency, memory, and **calibration** frontier of
compressed language models.

> Does compression degrade a model's calibration faster than it degrades its
> accuracy, and does that hold across compression families (post-training
> quantization, quantization-aware training, distillation)?

Accuracy, latency, and memory are the reference axes. Calibration (ECE, Brier,
reliability diagrams) is the contribution.

A model that ships to a phone or a cost-controlled endpoint has usually been
quantized or distilled first, and the check that authorizes the swap is an
accuracy comparison against the uncompressed model. If calibration degrades
faster than accuracy, that check passes while the deployed model's confidence has
stopped tracking its correctness, and every abstention threshold tuned against
the original is now set wrong.

Single 16GB GPU, roughly 20 GPU-hours, every path also runnable on a CPU laptop.
See `docs/architecture.md`.

## Status

Six rows banked on Qwen2.5-3B, 2000 MMLU items:

| variant | backend | accuracy | ECE |
|---|---|---|---|
| gguf-q8_0 | llama.cpp | 0.6485 | 0.2663 |
| fp16 | HF | 0.6470 | 0.2687 |
| gguf-q5_k_m | llama.cpp | 0.6430 | 0.2690 |
| int8-weightonly | HF (bnb) | 0.6365 | 0.2832 |
| int4-nf4 | HF (bnb) | 0.6235 | 0.2885 |
| gguf-q4_k_m | llama.cpp | 0.6210 | 0.2908 |

The FP16 baseline (95% CI [0.626, 0.667], Brier 0.601) matches Qwen's published
score near 65%. q8_0 and q5_k_m are lossless on both axes. At 4 bits ECE degrades
about twice as fast as accuracy in relative terms, with bitsandbytes NF4 and
llama.cpp q4_k_m agreeing independently. These are point estimates; the paired
bootstrap is pending.

Remaining: the vLLM batch (GPTQ, AWQ, W8A8, FP16 fidelity gate), paired bootstrap
CIs on every banked pair, and the QAT and distillation arms.

Test suite: 299 passing, 14 GPU-gated skips, under ruff and mypy strict.

## Setup

`uv`-managed, Python 3.11 pinned.

```bash
uv sync                      # base plus dev tools (ruff, mypy, pytest); CPU only
uv run pytest                # 299 pass, 14 GPU-gated skips; runs offline
uv sync --group oracles      # calibration-metric oracle libs (torchmetrics, netcal)
uv sync --group hf           # Track A HF backend (torch, transformers, bitsandbytes)
```

The Track B serving and quantization stack (vLLM, llmcompressor, torchao,
llama-cpp-python) pins against the real CUDA build and conflicts with the `hf`
group's transformers, so it installs on the pod by hand. See `docker/README.md`.

## Running

A run is one variant crossed with one eval profile, and appends one result row.
`mode` is a config field: `smoke` pins a tiny model and a 50-item CPU slice,
`full` uses the real models and eval sets on the pod.

```bash
# CPU smoke (downloads SmolLM2-135M)
FRONTIER_LIVE_MODELS=1 uv run --group hf \
  frontier run --config configs/variants/fp16.yaml --mode smoke

# full run on the pod
uv run --group hf frontier run --config configs/variants/int4-nf4.yaml --mode full

# frontier chart, reliability gallery, ECE-vs-bins sweep
uv run frontier plot --results results --plots-dir plots
```

`frontier run` also takes `--eval` (a profile under `configs/evals/`),
`--skip-latency`, and `--skip-predictions`. `frontier plot` takes
`--x memory|latency|cost_inv`, `--color-by family|track`, and `--figures`.
`frontier-quantize` produces the Track B GPTQ / AWQ / W8A8 / GGUF checkpoints and
is pod-only.

## Layout

```
configs/        one YAML per variant and per eval profile
  schema/       JSON Schema the configs validate against
  variants/     the compression matrix as data (fp16, int4-nf4, int4-gptq, ...)
  evals/        task profiles: primary-mmlu, secondary-arc, reasoning-cot
docs/           architecture and the result-row schema
src/frontier/   metrics, eval, backends, quantize, distill, latency, io, analysis
scripts/pod/    bootstrap, run a job, mirror results back, sync
tests/          unit tests; the calibration maths carry the heaviest coverage
results/        the result store (jsonl plus parquet); git-ignored
plots/          figures written by frontier plot; git-ignored
reports/        the technical writeup and its figures
```

## Documentation

- `docs/architecture.md` - two-track design, variant matrix, measurement rules, toolchain
- `docs/results_schema.md` - the result-row schema
- `docker/README.md` - the pod image, its version pins, and RunPod bring-up
- `scripts/pod/README.md` - the SSH job runner
- `tests/README.md` - what is covered and what is gated
