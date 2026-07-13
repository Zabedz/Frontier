# Frontier

A benchmark for the accuracy, latency, memory, and **calibration** frontier of
compressed language models. The question the project exists to answer:

> Does compression degrade a model's calibration faster than it degrades its
> accuracy, and does that hold across compression families (PTQ, QAT,
> distillation)?

Accuracy, latency, and memory are table stakes. Calibration (ECE, Brier,
reliability diagrams) is the contribution: a model that still scores well on an
accuracy dashboard but has lost track of when it is wrong is dangerous for any
system with an abstention or escalation path.

## Status

Scaffolding stage. Research and the work-package plan are done; experiment code
is not written yet. See `docs/architecture.md` for the architecture, variant
matrix, and conventions, `docs/methodology.md` for the measurement rules, and
`docs/decisions.md` for the running decision log.

## Layout

```
configs/        one YAML per variant; adding a variant is adding a file
  schema/       JSON Schema the configs are validated against
  variants/     the variant matrix, as data
docs/           methodology rules and the results-row schema
src/frontier/   the pipeline (metrics, eval, quantize, distill, latency, io)
tests/          unit tests; the calibration maths carry the heaviest coverage
results/        append-only parquet/jsonl result rows (the record)
```

## Two ways to run

The same code runs in two modes, chosen by config, never by a second codebase.

- **Smoke (local, CPU, no GPU):** a tiny model and a ~50-item eval slice. Proves
  the code path end to end on a laptop. `configs/smoke.yaml`.
- **Full (pod, 16GB GPU):** the real models and eval sets on RunPod.

```bash
# one-time setup (uv installs the toolchain from pyproject + uv.lock)
uv sync                      # local: CPU base + dev tools
uv sync --group oracles      # local: adds the calibration-metric oracle libs (torch)
uv sync --group gpu          # pod: adds the CUDA quantisation stack

# run one variant, produce one metric row
uv run frontier run --config configs/variants/fp16.yaml --mode smoke
```

## Hardware ceiling

Single GPU, 16GB VRAM maximum, ~20 GPU-hour total budget. Every design choice is
bound by this. See `docs/decisions.md` for why the teacher is capped at 3B and why
the distillation temperature axis is recovered post-hoc rather than by retraining.
