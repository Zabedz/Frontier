# Architecture

Project context and conventions. The result-row contract is in
`docs/results_schema.md`; the pod image and bring-up are in `docker/README.md`.

## What this is

> Does compression degrade calibration faster than it degrades accuracy, and does
> that hold across PTQ, QAT, and distillation?

Calibration is the contribution; accuracy, latency, and memory are the reference
axes. The output is a reproducible pipeline plus a short technical report ending
in a defensible "what I would deploy and why".

## Hard constraints

- **Single GPU, 16GB VRAM.** RunPod, 4090 / A4000 / L4 class.
- **~20 GPU-hour total budget.** Bias every choice toward cheap iteration.
- **50GB system RAM** on the pod.
- **Local dev has little or no GPU.** Every path runs in a CPU smoke mode (tiny
  model, ~50-item eval slice), selected by config.

## Scope discipline

Task and hardware are fixed; only the compression method varies. One task done
properly beats four done shallowly, so a spare slot goes to a distillation variant
before a second benchmark. A third base model is a stop-and-ask.

## Layout

Config-driven. One YAML in `configs/variants/` defines the compression; one YAML
in `configs/evals/` defines the task (primary MMLU, secondary ARC, reasoning CoT).
A run is (variant x eval-profile) and appends one result row. Adding a variant or a
task means adding a file. The reasoning profile runs on a representative subset of
variants, since it needs generation and is GPU-only.

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
- `backends/`   inference backends satisfying the eval logit seam (HF Track-A now;
                vLLM, llama.cpp, torchao later). Model loading lives here.
- `quantize/`   PTQ producers (llm-compressor GPTQ/AWQ, bnb, torchao, GGUF) and QAT.
- `distill/`    offline teacher top-k cache + student training (KD losses).
- `latency/`    TTFT / inter-token timing, memory + machine-state capture.
- `pipeline/`   the CLI and the config-to-row runner.
- `io/`         result-store read/write, provenance stamping.

## The two-track design

Compression method maps to a fixed inference backend, and those backends differ in
latency behaviour and logit access. A llama.cpp tokens/sec never shares a column
with a vLLM or HF number.

- **Track A (PyTorch/HF):** FP16, bnb NF4, bnb LLM.int8, torchao int8 weight-only.
  One `transformers` eval path, so logits come out the same way. The calibration
  core lives here, where logit access is directly comparable.
- **Track B (native serving):** W8A8 and INT4 (GPTQ/AWQ) on vLLM, GGUF k-quants on
  llama.cpp with full CUDA offload (`-ngl 99`, verified zero CPU spill). Its own
  latency track. A Track B method entering the calibration analysis must first
  reproduce the Track A baseline ECE from its FP16/Q8_0 baseline.

`backend` is a required field on every result row (`docs/results_schema.md`).

## Variant matrix

`cal` = calibration-set-as-variable axis;
`T` = track. Each variant is scored on the primary MMLU profile and the secondary
ARC profile; the six-variant reasoning subset also runs the GPQA CoT profile.

| variant                 | family   | backend        | T | notes |
|-------------------------|----------|----------------|---|-------|
| fp16                    | baseline | HF             | A | the reference point |
| int8-weightonly         | ptq      | HF (bnb/torchao)| A | memory win, flat speed |
| int8-w8a8               | ptq      | vLLM (llmc)    | B | real int8 compute; SmoothQuant |
| int4-gptq               | ptq      | vLLM (llmc)    | B | cal: in-domain vs OOD |
| int4-gptq-ood           | ptq      | vLLM (llmc)    | B | OOD calibration corpus |
| int4-awq                | ptq      | vLLM (llmc)    | B | cal: in-domain vs OOD |
| int4-awq-ood            | ptq      | vLLM (llmc)    | B | OOD calibration corpus |
| int4-nf4                | ptq      | HF (bnb)       | A | the QLoRA base |
| gguf-q4_k_m             | ptq      | llama.cpp/CUDA | B | latency track |
| gguf-q5_k_m             | ptq      | llama.cpp/CUDA | B | latency track |
| gguf-q8_0               | ptq      | llama.cpp/CUDA | B | ~FP16; low priority |
| ptq-3bit-torchao        | ptq      | torchao        | A | matched control for QAT |
| qat-3bit-lora           | qat      | torchao        | A | teacher, QAT+LoRA |
| student-qat-3bit-full   | qat      | torchao        | A | 0.5B student, full-param QAT |
| distill-hardlabel       | distill  | HF             | A | sequence-KD baseline |
| distill-softlabel-topk  | distill  | HF             | A | offline top-k KL, k=64 |

Distil-then-quantise reuses the PTQ configs on the distilled student checkpoint.
Pruning (Wanda / magnitude 2:4) is out of scope.

## Metrics

- **Quality:** task accuracy (exact letter match); calibration = ECE (equal-width
  and equal-mass/ACE, over a bin-count sweep), Brier + Murphy decomposition (the
  bin-free primary), reliability diagrams; perplexity on a held-out corpus, kept
  to show its weak correlation with task score.
- **Latency:** TTFT and inter-token latency reported **separately** (prefill is
  compute-bound, decode is memory-bandwidth-bound; conflating them hides the
  mechanism). Throughput at batch 1/4/16. Batch 32 dropped: OOM-prone and
  uninformative for a 16GB FP16 3B.
- **Memory:** peak VRAM per batch size; weights on disk vs resident; KV-cache
  growth with context length.
- **Cost proxy:** tokens/sec per GB of VRAM, collapsing three axes into one
  headline chart.

## Measurement rules

Enforced in the result schema (`docs/results_schema.md`).

- **Bit-width is always paired with an implementation.** Every measurement records
  its backend. An INT4 number and an FP16 number in one latency column must share a
  backend, or they sit in different tracks.
- **Guard the confidence signal.** Three confounds move with the treatment and can
  masquerade as calibration change: letter selection bias (controlled by cyclic
  option-order permutation, with permutation robustness reported as its own number),
  MMLU label noise (~6.5%, up to 57% in some subjects, so the final ECE uses
  MMLU-Redux and reports raw-vs-redux sensitivity), and first-token vs
  generated-answer divergence (agreement validated on a sample).
- **Own the calibration maths, oracle-check them.** A silently-wrong ECE invalidates
  the whole project. `torchmetrics`, `netcal`, and `sklearn` are oracles on synthetic
  fixtures, and they agree only when binning scheme, bin count, and multiclass
  reduction match. `tests/README.md` has the fixtures. Report a bin-count sweep and
  an equal-mass ACE, and anchor the headline on the bin-free Brier reliability term.
- **Two clocks.** TTFT and ITL measured and reported separately, CUDA events with a
  single synchronize outside the decode loop, 2-5 warmup iterations discarded, >=20
  trials, median and p95.
- **Paired resampling.** Bootstrap delta-ECE and delta-accuracy on the same resample
  indices and require the CI to exclude zero. Eval sets run >=1000 items; below ~500
  the binning noise dominates.
- **Clocks are logged.** RunPod containers rarely hold the lock privilege, so each
  measurement carries its clock, temperature, and power reading, and drifting runs
  are flagged.
- **Calibration corpus is an axis.** GPTQ and AWQ each run with an in-domain and an
  out-of-domain corpus at matched sample count and seqlen.

## Toolchain (current as of mid-2026; re-verify before pinning)

AutoGPTQ (archived Apr 2025) and AutoAWQ (archived May 2025) are dead. torchtune
(KD + QAT+LoRA reference recipes) is maintenance-only; use it as reference.

- **PTQ INT4:** `llmcompressor` (GPTQ + AWQ, emits compressed-tensors, served by
  vLLM on the marlin kernels). `gptqmodel` as a secondary GPTQ cross-check.
- **4-bit / int8 weight-only:** `bitsandbytes` (NF4, LLM.int8), `torchao`.
- **QAT:** `torchao` QATConfig prepare/convert, so PTQ and QAT share one config
  and differ only in whether weights were adapted. Unsloth is an optional GPU
  accelerator; being GPU-only, it stays off the smoke path.
- **GGUF:** `llama.cpp` / `llama-cpp-python` built with CUDA, full offload.
- **Distillation:** offline top-k cache + a custom KD loss on HF Trainer / TRL.
  TRL GKD for the single online reference comparison.

The pod-only package list and bring-up are in `docker/README.md`.

## Conventions

- Package manager: `uv`. `uv.lock` is committed; the `pyproject.toml` bounds are
  floors.
- `ruff` (strict) and `mypy` (strict) via `pre-commit`. Everything typed.
- Calibration-metric oracle libraries live in the `oracles` group
  (`uv sync --group oracles`). The GPU stack is pod-only (`docker/README.md`).
- Smoke every path on CPU first. GPU time is for real runs.
