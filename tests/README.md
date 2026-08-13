# Tests

```
uv run pytest
```

299 passing, 14 gated skips. Suites mirror `src/frontier/`: `metrics/`, `eval/`,
`backends/`, `quantize/`, `latency/`, `pipeline/`, `io/`, `analysis/`. The calibration
maths carry the heaviest coverage; a silently-wrong ECE would invalidate the project.

## Calibration core (`metrics/`)

| Group | What it pins |
| --- | --- |
| Analytic fixtures | all-confident-correct gives ECE 0, all-confident-wrong gives ECE 1, a two-bin case computed by hand, a large calibrated sample near the finite-sample bin bias |
| Oracle cross-checks | agreement with `torchmetrics.MulticlassCalibrationError` (equal-width), `netcal` ECE (equal-width) and ACE (equal-mass), and a hand-built `sklearn.calibration_curve` reduction, asserted only once binning scheme, bin count, and multiclass reduction are matched |
| Brier + Murphy | reliability - resolution + uncertainty reconstructs the total Brier score |
| Bin-count sweep | monotonic behaviour, and equal-mass vs equal-width bias direction on skewed confidence |
| Paired bootstrap | both arrays index with one resample vector, the regression against the independent-resampling bug; CI width shrinks with n; a calibrated large sample has a delta-ECE CI containing zero |

## Contracts and config

- Every `configs/variants/*.yaml` validates against `configs/schema/variant.schema.json`.
- A `ResultRow` cannot be constructed without provenance and backend
  (`test_schema_contract.py`).
- Resume-skip keys on config_hash + seed + task (`io/test_store.py`).

## Backends and latency

Provider fakes drive the eval seam on CPU, so `backends/` and `latency/` cover control
flow without a GPU:

- the vLLM full-vocab logprob read
- the llama.cpp last-position read
- the native probes' server lifecycle and command construction
- the `vllm bench` / `llama-bench` JSON parsers against captured payloads

## Gated tests

Skipped by default, and named `*_live.py` where they download.

| Gate | Unlocks |
| --- | --- |
| `FRONTIER_LIVE_MODELS=1` | anything that pulls a checkpoint: `test_hf_live.py`, `test_smoke_end_to_end.py`, `test_timing_live.py` |
| `FRONTIER_LIVE_DATASETS=1` | the dataset loaders |
| a CUDA device | `test_timing_live.py` |
| `vllm` importable | `test_vllm_fidelity.py` |
| `llama_cpp` importable | `test_llama_cpp_live.py` |
| `llmcompressor` importable | `test_producers_live.py`, `test_recipes.py` |

The end-to-end smoke runs the full config-to-row path on SmolLM2-135M over a ~50-item
slice and appends one valid row with every provenance and backend field populated.
