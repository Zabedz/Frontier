# Tests

299 passing, 14 gated skips. The calibration maths carry the heaviest coverage: a
silently-wrong ECE would invalidate the whole project.

Suites mirror `src/frontier/`: `metrics/`, `eval/`, `backends/`, `quantize/`,
`latency/`, `pipeline/`, `io/`, `analysis/`.

## Calibration core (`metrics/`)

- **Analytic fixtures.** all-confident-correct gives ECE 0; all-confident-wrong gives
  ECE 1; a two-bin case computed by hand; a large perfectly-calibrated sample lands
  near the finite-sample bin bias.
- **Oracle cross-checks on synthetic data.** Our ECE agrees with
  `torchmetrics.MulticlassCalibrationError` (equal-width), `netcal` ECE (equal-width)
  and ACE (equal-mass), and a hand-built `sklearn.calibration_curve` reduction, once
  binning scheme, bin count, and multiclass reduction are matched. Mismatched binning
  is expected to disagree, and the test asserts agreement only under matched settings.
- **Brier + Murphy decomposition.** reliability - resolution + uncertainty
  reconstructs the total Brier score.
- **Bin-count sweep** monotonic behaviour, and equal-mass vs equal-width bias direction
  on a skewed-confidence fixture.
- **Paired bootstrap.** Both arrays index with the same resample indices; this is a
  regression test against the independent-resampling bug. CI width shrinks with n, and
  a known-calibrated large sample has a delta-ECE CI containing zero.

## Contracts and config

- Every `configs/variants/*.yaml` validates against
  `configs/schema/variant.schema.json`.
- A `ResultRow` cannot be constructed without provenance and backend
  (`test_schema_contract.py`).
- Resume-skip keys on config_hash + seed + task (`io/test_store.py`).

## Backends and latency

Provider fakes drive the eval seam on CPU, so `backends/` and `latency/` cover control
flow without a GPU: the vLLM full-vocab logprob read, the llama.cpp last-position
read, the native probes' server lifecycle and command construction, and the
`vllm bench` / `llama-bench` JSON parsers against captured payloads.

## Gated tests

Live and GPU paths are skipped by default and named `*_live.py` where they download:

- `FRONTIER_LIVE_MODELS=1` for anything that pulls a checkpoint (`test_hf_live.py`,
  `test_smoke_end_to_end.py`, `test_timing_live.py`).
- `FRONTIER_LIVE_DATASETS=1` for the dataset loaders.
- Import-gated on the pod stack: `test_vllm_fidelity.py` (vllm),
  `test_llama_cpp_live.py` (llama_cpp), `test_producers_live.py` and `test_recipes.py`
  (llmcompressor).
- `test_timing_live.py` also needs a CUDA device.

The end-to-end smoke runs the full config-to-row path on SmolLM2-135M over a ~50-item
slice and appends one valid row with every provenance and backend field populated.
