# Methodology

The rules that separate this from a blog post. They are enforced in code and in
the result schema, not left to good intentions. This file is the rationale; the
enforcement points are noted against each.

## 1. implementation x bit-width, never bit-width in the abstract

The most common error in amateur quantisation benchmarks is reporting "INT4 is
slow" when the truth is "this dequant kernel is slow". Every measurement records
its backend (`Backend` record, required). An INT4 number and an FP16 number in the
same latency column must share a backend, or they sit in different tracks. See the
two-track design in `docs/architecture.md`.

## 2. Calibration is the instrument, so guard the instrument

The confidence signal is a softmax over answer-letter logprobs on a multiple-choice
task. Three confounds are compression-sensitive, meaning they move with the
treatment variable and can masquerade as calibration change:

- **Letter selection bias** (models favour certain letters / positions). Control
  it: score each item under cyclic option-order permutation (PriDe-style), debias
  the letter distribution before computing confidence, and report calibration
  robustness to permutation as its own number.
- **Label noise** (MMLU ~6.5% errors, up to 57% in some subjects). Compute the
  final ECE on de-noised labels (MMLU-Redux) and report raw-vs-redux sensitivity.
- **First-token vs generated-answer divergence** (the argmax letter may disagree
  with what the model would actually write). Validate agreement on a sample; if it
  is high, the letter softmax is measuring the wrong quantity.

## 3. Calibration maths: own it, oracle-check it

Own the ECE/Brier implementation and cross-validate against three libraries on
synthetic fixtures (`torchmetrics`, `netcal`, `sklearn.calibration_curve`). They
agree only when binning scheme, bin count, and multiclass reduction match; making
them agree is the test. Analytic fixtures with known answers pin the corners.
Report ECE over a bin-count sweep and an equal-mass ACE; anchor the headline on the
bin-free Brier reliability term. Never publish a single equal-width ECE at one bin
count (it is biased and bin-sensitive, and can invert the conclusion).

## 4. Latency: two clocks, not one

- **TTFT** (prefill, compute-bound) and **inter-token latency** (decode,
  memory-bandwidth-bound) are measured and reported separately.
- `torch.cuda.Event(enable_timing=True)` pairs, one `torch.cuda.synchronize()`
  before reading `elapsed_time()`. Do not put a synchronize inside the per-token
  loop; it serialises and inflates fast steps.
- Discard the first 2-5 iterations (warmup). >=20 trials. Median and p95, never
  mean.
- Track-B peak VRAM semantics: numbers are read while the workload is alive, at the
  reference context length only (the analytic `kv_cache_mb` column carries the context
  scaling). For vLLM, "peak" is the engine's preallocated reservation, so
  `tok_s_per_gb`'s denominator is the serving reservation and variants separate on the
  throughput numerator; for llama.cpp it is the max of a 0.5s-interval sample of
  `nvidia-smi` while `llama-bench` runs.
- The vLLM reservation is the card minus a fixed headroom, capped at 0.92 and floored at
  a 15GB card (`vllm_memory_fraction`, `latency/native.py`). The eval engine and the
  serve command call the same helper, so the reservation the row records is the one the
  eval ran under. A larger pod therefore buys a larger KV cache and more throughput.
- The consequence: a vLLM row's `peak_vram_mb` and `tok_s_per_gb` are only comparable
  against other vLLM rows measured on the same card, and `peak_vram_mb` does not separate
  vLLM variants from each other at all, because the engine takes whatever it is given.
  For vLLM the variant's memory signal is `weights_resident_mb` and `weights_disk_mb`,
  which come from the checkpoint and do separate int4 from int8 from fp16. Run every
  vLLM row on one card and report `tok_s_per_gb` within that card.

## 5. Quiet machine, honestly

Try `nvidia-smi -pm 1`, `--lock-gpu-clocks`, `--lock-memory-clocks`. Assume it
fails on RunPod (containers rarely hold the privilege; expect "Insufficient
Permissions"). Fallback: log `clocks.sm`, `clocks.mem`, `temperature.gpu`,
`power.draw` per measurement, warm to a steady thermal state, repeat many times,
report the observed clock range, and flag runs whose clocks drifted. Two models are
only comparable when measured back-to-back under the same logged clock state.

## 6. Statistical honesty

- Bootstrap CI on every accuracy and every ECE (`scipy.stats.bootstrap`).
- **Paired** resampling: resample row indices, then index into both confidences and
  labels. Never let the two arrays resample independently.
- The headline claim (compression hurts calibration more than accuracy) is a claim
  about a *difference*. Bootstrap delta-ECE and delta-accuracy on the same
  resamples and require the CI to exclude zero.
- >=3 seeds for anything with training or quantisation-calibration randomness.
- Eval set >=1000 items (a few thousand preferred) for stable 10-bin ECE. Below
  ~500 the binning noise dominates. n=1000 gives ~1.5% accuracy SE, so accuracy
  gaps under ~3-4 points are noise. Trust the bootstrap CI over eyeballing a table.

## 7. Calibration-set-as-variable

PTQ methods pick scales from a small calibration corpus. That corpus is an explicit
variable, not a footnote: run GPTQ and AWQ with an in-domain corpus and an
out-of-domain corpus at matched sample count and seqlen. Pre-registered hypothesis:
AWQ derives per-channel scales from calibration *activations*, so an out-of-domain
corpus should perturb AWQ's scales (and its ECE) more than GPTQ's, whose objective
only minimises per-layer output reconstruction error. Hold the corpus, sample
count, and seqlen identical across the two methods, or the axis is confounded.
