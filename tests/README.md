# Tests

The calibration maths carry the heaviest coverage: a silently-wrong ECE would
invalidate the whole project, so it is tested before it is trusted. No experiment
code exists yet; this is the test plan the first implementation session works to.

## Calibration core (unit, CPU, must land first)

- **Analytic fixtures with known answers.** all-confident-correct -> ECE 0;
  all-confident-wrong -> ECE 1; a two-bin case computed by hand; a large
  perfectly-calibrated sample -> ECE near the finite-sample bin bias, not exactly 0.
- **Oracle cross-checks on synthetic data.** Our ECE agrees with
  `torchmetrics.MulticlassCalibrationError` (equal-width) and with `netcal` ECE
  (equal-width) and ACE (equal-mass), and a hand-built `sklearn.calibration_curve`
  reduction, once binning scheme, bin count, and multiclass reduction are matched.
  Mismatched binning is expected to disagree; the test asserts agreement only under
  matched settings.
- **Brier + Murphy decomposition.** reliability - resolution + uncertainty
  reconstructs the total Brier score.
- **Bin-count sweep monotonic behaviour** and equal-mass vs equal-width bias
  direction on a skewed-confidence fixture.

## Bootstrap CIs (unit, CPU)

- Paired resampling indexes both arrays with the same indices (a regression test
  that guards against the independent-resampling bug).
- CI width shrinks with n; a known-calibrated large sample has a delta-ECE CI that
  contains zero.

## Config + schema (unit, CPU)

- Every `configs/variants/*.yaml` validates against
  `configs/schema/variant.schema.json`.
- A `ResultRow` cannot be constructed without provenance and backend.

## Smoke (integration, CPU, marked slow)

- `configs/smoke.yaml` runs the full config-to-row path on SmolLM2-135M over a
  ~50-item slice and appends one valid row with all provenance and backend fields
  populated.

GPU-only paths (marlin INT4 decode, Unsloth, vLLM serving) are marked `gpu` and
skipped in the local CPU loop; they run on the pod.
