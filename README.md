# Frontier

A benchmark for the accuracy, latency, memory, and **calibration** frontier of
compressed language models. The question the project exists to answer:

> Does compression degrade a model's calibration faster than it degrades its
> accuracy, and does that hold across compression families (post-training
> quantization, quantization-aware training, distillation)?

Accuracy, latency, and memory are the reference axes. Calibration (ECE, Brier,
reliability diagrams) is the contribution. A model that still scores well on an
accuracy dashboard but has lost track of when it is wrong is dangerous for any
system with an abstention or escalation path, and that failure does not show up
in an accuracy number.

The whole thing runs on a single 16GB GPU inside a budget of roughly 20
GPU-hours, and every code path also runs on a laptop with no GPU. Those two
limits shape every design choice; see `docs/architecture.md` for the reasoning.

## Status

The pipeline is built, reviewed, and tested: the calibration metrics (ECE
variants, Brier plus its decomposition, reliability, paired bootstrap CIs), the
eval core (MMLU / MMLU-Redux / ARC loaders, answer-letter softmax, cyclic
permutation debiasing), the Hugging Face inference backend, the append-only
result store, the latency and memory rig, and the analysis and plotting code.
The test suite is 283 passing with 14 GPU-gated skips, under ruff and mypy
strict.

The first real result is banked. FP16 Qwen2.5-3B on 2000 MMLU items gives
accuracy 0.647 (95% CI [0.626, 0.667]), ECE 0.269, and Brier 0.601, which lines
up with Qwen's published score near 65%. The instrument reproduces a known
number, so it is sound.

The remaining work is the compressed-variant batch (the bitsandbytes NF4 and
int8 weight-only variants run on the current Hugging Face path now; the GPTQ,
AWQ, W8A8, and GGUF variants run on the pod serving stack) and the two training
arms (quantization-aware training and distillation). See `SESSION_STATE.md` for
the current pod state and the exact next steps.

## Setup

The project is managed with `uv`. `uv.lock` is the reproducibility source of
truth; the version bounds in `pyproject.toml` are only the floor. Python is
pinned to 3.11.

```bash
uv sync                      # base runtime plus dev tools (ruff, mypy, pytest); CPU only
uv run pytest                # 283 pass, 14 GPU-gated skips; runs offline on a laptop

uv sync --group oracles      # adds the calibration-metric oracle libs (torchmetrics, netcal)
uv sync --group hf           # adds the Track A Hugging Face backend (torch, transformers, bitsandbytes)
```

The Track B serving and quantization stack (vLLM, llmcompressor, torchao,
llama-cpp-python) is not a locked dependency group. It pins against the real
CUDA build and its transformers version conflicts with the `hf` group, so it is
installed on the pod by hand. The versions and install order are in
`docs/pod-setup.md`.

## Running

A run is one variant crossed with one eval profile, and it appends one result
row. Adding a variant or a task means adding a YAML file under `configs/`. Mode
is a config field resolved by the runner: `smoke` pins a tiny model and a 50-item
slice on CPU, `full` uses the real models and eval sets on the pod.

```bash
# CPU smoke: proves the end-to-end path on a laptop (downloads SmolLM2-135M)
FRONTIER_LIVE_MODELS=1 uv run --group hf \
  frontier run --config configs/variants/fp16.yaml --mode smoke

# a full run on the pod: score one variant, append its row(s)
uv run --group hf frontier run --config configs/variants/int4-nf4.yaml --mode full

# read the result store and draw the frontier chart, reliability gallery, and ECE-vs-bins sweep
uv run frontier plot --results results --plots-dir plots
```

`frontier run` takes `--config`, `--eval` (an eval profile under
`configs/evals/`), `--mode smoke|full`, and `--skip-latency` /
`--skip-predictions` to trim what it records. `frontier plot` takes `--x
memory|latency|cost_inv`, `--color-by family|track`, and `--figures`. A second
entry point, `frontier-quantize`, produces the Track B GPTQ / AWQ / W8A8 / GGUF
checkpoints and runs on the pod only.

## Layout

```
configs/        one YAML per variant and per eval profile; a run is variant x profile
  schema/       JSON Schema the configs are validated against
  variants/     the compression matrix, as data (fp16, int4-nf4, int4-gptq, ...)
  evals/        task profiles: primary-mmlu, secondary-arc, reasoning-cot
docs/           architecture, methodology, decisions, results schema, pod setup
src/frontier/   the pipeline: metrics, eval, backends, quantize, distill, latency, io, analysis
scripts/pod/    the pod scripts: bootstrap, run a job, mirror results back, sync
tests/          unit tests; the calibration maths carry the heaviest coverage
results/        the result store (jsonl plus parquet); regenerated, git-ignored
plots/          figures written by frontier plot; regenerated, git-ignored
reports/        the technical writeup, with the headline figures committed alongside it
```

`results/` and `plots/` are rebuilt from the pipeline, so they are not committed.
Every result row carries the git SHA, the resolved config hash, the model
revision, and the seed, so a run reconstructs from its config and that
provenance alone.

## Documentation

The detail lives in `docs/`: `architecture.md` (the two-track design, the
variant matrix, the toolchain), `methodology.md` (the measurement rules that
make the numbers trustworthy), `decisions.md` (the running decision log),
`results_schema.md` (the result-row schema), and `pod-setup.md` (the GPU pod
stack and bring-up).
