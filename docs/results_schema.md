# Result-row schema

Every variant run appends exactly one row to the result store (`results/`, parquet
plus a jsonl mirror). Analysis and plotting read this store; nothing recomputes from
raw model outputs. The typed contract is `ResultRow` and its nested records in
`src/frontier/schema.py`.

`Backend` and `Provenance` are required, non-default fields, so a row cannot be
constructed without recording which kernel produced the number.

## Provenance (required)

| field            | type   | note |
|------------------|--------|------|
| git_sha          | str    | commit the run was produced from |
| config_hash      | str    | hash of the resolved variant config |
| model_id         | str    | e.g. Qwen/Qwen2.5-3B-Instruct |
| model_revision   | str    | HF revision / commit pin |
| hardware_id      | str    | GPU model + pod id |
| driver_version   | str    | NVIDIA driver |
| cuda_version     | str    | CUDA toolkit / runtime |
| seed             | int    | |
| timestamp        | str    | ISO 8601, UTC |

## Backend (required, first-class)

| field             | type | note |
|-------------------|------|------|
| inference_backend | str  | hf, vllm, llama_cpp, torchao |
| backend_version   | str  | pinned package version |
| weight_dtype      | str  | fp16, int8, int4, nf4, q4_k_m, int3 |
| kv_cache_dtype    | str  | fp16, int8, ... |
| gpu_offload_layers| int  | llama.cpp -ngl; must equal layer count for a clean GPU number |
| track             | str  | A (PyTorch/HF) or B (native serving) |

## Variant

variant_name, family (baseline/ptq/qat/distill), method, bit_width, group_size,
calibration_corpus (none/in_domain/ood), calibration_samples.

## Task

task_name, split, num_items, prompt_style (zeroshot/fiveshot), scoring
(letter_softmax/acc_norm), permutation_scheme (none/cyclic), labels (raw/redux),
cot (bool).

## Quality

accuracy (+ ci_low, ci_high), ece_equal_width, ece_equal_mass_ace, ece_bin_sweep
(map of bin_count -> ece), brier, brier_reliability, brier_resolution,
brier_uncertainty, nll, perplexity, temperature_scaled (bool), temperature.
Calibration numbers carry bootstrap CIs; deltas across variants are bootstrapped
paired on the same resamples.

## Latency (per batch size)

batch_size, ttft_median_ms, ttft_p95_ms, itl_median_ms, itl_p95_ms,
throughput_tok_s, n_trials, warmup_discarded. TTFT and ITL are always separate.

## Memory (per batch size / context length)

peak_vram_mb, weights_disk_mb, weights_resident_mb, kv_cache_mb, context_len.

`peak_vram_mb` is read while the workload is alive, at the reference context length
only; the analytic `kv_cache_mb` column carries the context scaling. Its meaning
depends on the backend. HF reads the CUDA allocator, llama.cpp takes the max of a
0.5s-interval `nvidia-smi` sample around `llama-bench`, and vLLM reports the engine's
preallocated reservation. That reservation is sized to the card
(`vllm_memory_fraction`, `latency/native.py`), so a vLLM row's `peak_vram_mb` and
`tok_s_per_gb` are comparable only against vLLM rows on the same card, and every vLLM
row runs on one card. `peak_vram_mb` does not separate vLLM variants from each other,
since the engine takes whatever it is given; for vLLM the variant's memory signal is
`weights_resident_mb` and `weights_disk_mb`.

## Machine state (captured per measurement)

gpu_clock_sm_mhz, gpu_clock_mem_mhz, gpu_temp_c, power_w, clocks_locked (bool),
clock_drift_flag (bool). If clocks could not be locked (the expected RunPod case),
`clocks_locked` is false and the observed clock range is what defends the number.

## Cost proxy

tok_s_per_gb = throughput_tok_s / peak_vram_gb. The headline single number.
