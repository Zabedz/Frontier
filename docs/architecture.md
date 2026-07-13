# Architecture

Project context and conventions for Frontier. `docs/methodology.md` holds the
measurement rules; `docs/decisions.md` holds the running decision log (why things
are the way they are); `docs/pod-setup.md` holds the GPU pod stack and bring-up.

## What this is

A benchmark characterising the accuracy x latency x memory x calibration frontier
of a family of compressed language model variants. The central research question:

> Does compression degrade calibration faster than it degrades accuracy, and does
> that hold across PTQ, QAT, and distillation?

Calibration is the contribution. Accuracy, latency, and memory are the reference
axes everything is measured against. The output is a reproducible pipeline plus a
short technical report ending in a defensible "what I would deploy and why".

## Hard constraints (do not design around these being soft)

- **Single GPU, 16GB VRAM maximum.** RunPod, 4090 / A4000 / L4 class.
- **~20 GPU-hour total budget.** A design that needs 40 GPU-hours is a failed
  design. Bias every choice toward cheap iteration.
- **50GB system RAM** on the pod.
- **Local dev has little or no GPU.** Everything must run in a CPU smoke mode
  (tiny model, ~50-item eval slice), selected by config, not a second codebase.

## Scope discipline

The failure mode is combinatorial explosion. Task and hardware are fixed; only the
compression method varies. One task done properly beats four done shallowly. If
the urge is to add a benchmark, add a distillation variant instead. A third base
model is a stop-and-ask, not a default.

## Layout

Config-driven. One YAML in `configs/variants/` defines the compression; one YAML
in `configs/evals/` defines the task (primary MMLU, secondary ARC, reasoning CoT).
A run is (variant x eval-profile) and appends one result row. Adding a variant, or
a task, means adding a file, never editing code. The reasoning profile is run only
on a representative subset of variants (it needs generation and is GPU-only).

```
config (YAML)  ->  pipeline  ->  result row (parquet/jsonl, append-only)
                                        |
                        analysis + plotting read the store; nothing recomputes
```

Package layout under `src/frontier/`:

- `schema.py`   typed contracts: `VariantConfig` and `ResultRow` (+ nested
                `Provenance`, `Backend`, `TaskSpec`, `Quality`, `Latency`,
                `Memory`, `MachineState`). Frozen dataclasses, no behaviour.
- `metrics/`    calibration (ECE variants, Brier + decomposition, reliability),
                bootstrap CIs, perplexity. Pure CPU. Heaviest test coverage.
- `eval/`       task loaders, prompt building, confidence extraction, correctness.
- `quantize/`   PTQ producers (llm-compressor GPTQ/AWQ, bnb, torchao, GGUF) and QAT.
- `distill/`    offline teacher top-k cache + student training (KD losses).
- `latency/`    TTFT / inter-token timing, memory + machine-state capture.
- `pipeline/`   the CLI and the config-to-row runner.
- `io/`         result-store read/write, provenance stamping.

## The two-track design (a fairness rule, not a convenience)

Compression method maps to a fixed inference backend, and those backends are not
interchangeable for latency or for logit access. Never compare a llama.cpp
tokens/sec against a vLLM or HF number in the same column.

- **Track A (PyTorch/HF):** FP16, bnb NF4, bnb LLM.int8, torchao int8 weight-only.
  One `transformers` eval path, so logits come out the same way. **The calibration
  core lives here** because logit access is directly comparable.
- **Track B (native serving):** W8A8 and INT4 (GPTQ/AWQ) on vLLM, GGUF k-quants on
  llama.cpp with full CUDA offload (`-ngl 99`, verified zero CPU spill). Reported
  as its own latency track. If a Track B method enters the calibration analysis,
  its FP16/Q8_0 baseline must first reproduce the Track A baseline ECE.

`backend` is a first-class field on every result row (see `docs/results_schema.md`).
The row cannot be constructed without it.

## Variant matrix

Settled (see `docs/decisions.md`). `cal` = calibration-set-as-variable axis;
`T` = track. Each variant is scored on the primary MMLU profile and the secondary
ARC profile; the six-variant reasoning subset is also scored on the GPQA CoT profile.

| variant                 | family   | backend        | T | notes |
|-------------------------|----------|----------------|---|-------|
| fp16                    | baseline | HF             | A | the reference point |
| int8-weightonly         | ptq      | HF (bnb/torchao)| A | memory, not speed |
| int8-w8a8               | ptq      | vLLM (llmc)    | B | real int8 compute; SmoothQuant |
| int4-gptq               | ptq      | vLLM (llmc)    | B | cal: in-domain vs OOD |
| int4-gptq-ood           | ptq      | vLLM (llmc)    | B | OOD calibration corpus |
| int4-awq                | ptq      | vLLM (llmc)    | B | cal: in-domain vs OOD |
| int4-awq-ood            | ptq      | vLLM (llmc)    | B | OOD calibration corpus |
| int4-nf4                | ptq      | HF (bnb)       | A | the QLoRA base |
| gguf-q4_k_m             | ptq      | llama.cpp/CUDA | B | latency track |
| gguf-q5_k_m             | ptq      | llama.cpp/CUDA | B | latency track |
| gguf-q8_0               | ptq      | llama.cpp/CUDA | B | ~FP16; low priority, likely cut |
| ptq-3bit-torchao        | ptq      | torchao        | A | matched control for QAT |
| qat-3bit-lora           | qat      | torchao        | A | teacher, QAT+LoRA |
| student-qat-3bit-full   | qat      | torchao        | A | 0.5B student, full-param QAT |
| distill-hardlabel       | distill  | HF             | A | sequence-KD baseline |
| distill-softlabel-topk  | distill  | HF             | A | offline top-k KL, k=64 |

Distil-then-quantise reuses the PTQ configs on the distilled student checkpoint.
Pruning (Wanda / magnitude 2:4) is out of scope and assumed cut.

## Metrics

- **Quality:** task accuracy (exact letter match); calibration = ECE (equal-width
  and equal-mass/ACE, reported over a bin-count sweep), Brier + Murphy
  decomposition (the bin-free primary), reliability diagrams; perplexity on a
  held-out corpus, kept specifically to show its weak correlation with task score.
- **Latency:** TTFT and inter-token latency reported **separately** (prefill is
  compute-bound, decode is memory-bandwidth-bound; conflating them hides the
  mechanism). Throughput at batch 1/4/16 (32 dropped, OOM-prone and uninformative
  for a 16GB FP16 3B). Discard warmup, >=20 trials, report median and p95, never
  mean.
- **Memory:** peak VRAM per batch size; weights on disk vs resident; KV-cache
  growth with context length.
- **Cost proxy:** tokens/sec per GB of VRAM. The number a deployment engineer
  actually uses; collapses three axes into one headline chart.

The measurement rules that make these numbers trustworthy are in
`docs/methodology.md` and are enforced in the result schema, not left to intentions.

## The calibration maths get the most tests

A silently-wrong ECE invalidates the whole project. Own the implementation; treat
libraries as oracles, not as the implementation.

- Write our own binned ECE (top-label confidence = max softmax; equal-width bins;
  L1 weighted by bin mass) and cross-validate on synthetic fixtures against
  `torchmetrics` MulticlassCalibrationError (equal-width), `netcal` ECE/ACE
  (equal-width + adaptive), and a hand-built `sklearn.calibration_curve` check.
  They agree only when binning scheme, bin count, and multiclass reduction match
  exactly; making them agree on fixtures **is** the test.
- Analytic fixtures with known answers: all-confident-correct -> ECE 0;
  all-confident-wrong -> ECE 1; a two-bin hand-computed case; a large
  perfectly-calibrated sample -> ECE near the finite-sample bin bias, not 0.
- Never publish a single ECE: report the bin-count sweep, an equal-mass ACE, and
  anchor the headline on the bin-free Brier reliability term.

## Toolchain (current as of mid-2026; re-verify before pinning)

AutoGPTQ (archived Apr 2025) and AutoAWQ (archived May 2025) are dead; do not
build on them. torchtune (KD + QAT+LoRA reference recipes) is maintenance-only;
use it as reference, not a live dependency.

- **PTQ INT4:** `llmcompressor` (GPTQ + AWQ, emits compressed-tensors, served by
  vLLM on the marlin kernels). `gptqmodel` as a secondary GPTQ cross-check.
- **4-bit / int8 weight-only:** `bitsandbytes` (NF4, LLM.int8), `torchao`.
- **QAT:** `torchao` QATConfig prepare/convert (PTQ and QAT share one config, so
  the only difference between the arms is whether weights were adapted). Unsloth as
  an optional GPU accelerator, never the smoke-test backbone (it is GPU-only).
- **GGUF:** `llama.cpp` / `llama-cpp-python` built with CUDA, full offload.
- **Distillation:** offline top-k cache + a custom KD loss on HF Trainer / TRL.
  TRL GKD only for the single online reference comparison.

The concrete pod-only package list and bring-up are in `docs/pod-setup.md`.

## Running smoke vs full

Mode is a config field, resolved by the runner. Smoke pins a tiny model
(SmolLM2-135M by default), a ~50-item eval slice, CPU, and skips GPU-only kernels
(marlin INT4 decode, Unsloth). Full uses the real models and eval sets on the pod.
Smoke every path on CPU first; GPU time is for real runs only.

## Conventions

- Package manager: `uv`. `uv.lock` is committed and is the reproducibility source
  of truth; version bounds in `pyproject.toml` are only the floor.
- `ruff` (strict) and `mypy` (strict) via `pre-commit`. Everything typed.
- The calibration-metric oracle libraries live in the `oracles` dependency group
  (`uv sync --group oracles`); the GPU stack is documented in `docs/pod-setup.md`
  and added to `pyproject.toml` at pod-provision time, pinned against the real CUDA
  environment.
