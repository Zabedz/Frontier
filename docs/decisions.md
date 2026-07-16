# Decision log

Newest first. Each entry: the decision, the reasoning, and whether it is settled
or provisional. Kept current as the project evolves; this is the memory of why
things are the way they are.

### 2026-07-16 Track B runs in a separate pod venv from Track A (SETTLED)
The two tracks pin different, non-overlapping transformers versions and cannot share one
environment. Track A's HF backend tracks the latest transformers 5.x (the `dtype=` API);
the Track-B serving/quantise stack caps transformers at `<=5.10.1` (llmcompressor's
ceiling) and vLLM fixes its own torch build. So the pod holds two venvs. venv-A is
`uv sync --group hf` (latest transformers 5.x, torch, accelerate, and bitsandbytes behind a
linux marker) and runs the fp16 and bnb `nf4`/`int8` variants through the HF backend.
venv-B is a separate pod-only install on the local disk (vllm, llmcompressor,
compressed-tensors, llama-cpp-python built with CUDA, gptqmodel, torchao; no bitsandbytes),
pinned as one uv resolution pass at transformers 5.10.1 with a compressed-tensors override,
and runs `frontier-quantize` and the `vllm`/`llama_cpp` `frontier run`. The single-pass pin
is load-bearing: a split install lets vLLM's unbounded `transformers>=5.5.3` pull a newer
transformers, which drops llmcompressor to its transformers-4 line and fails with
`Could not import module 'PreTrainedModel'` (the first pod hit exactly this). The batch
driver selects the venv per variant by `backend.inference_backend`. The Track-B stack is
deliberately not a locked dependency group, so a laptop `uv lock`/`uv sync` never tries to
resolve vLLM and stays CPU-clean; venv-B's versions pin on the pod and in the pre-baked
image (see `docker/`). This replaces the earlier single-`gpu`-group sketch, which was
self-contradictory (it listed transformers>=5.0 alongside llmcompressor, which cannot
co-resolve).

### 2026-07-16 GATE: the vLLM latency probe is unvalidated, fix it on the pod (OPEN)
`frontier.latency.native.NativeVllmLatency` is not yet correct and must be fixed against
real vLLM on the pod before any Track-B vLLM latency number is recorded. Two faults need a
GPU to settle, so they are not guessed blind. First, it benchmarks the wrong artifact: the
bench command points at `variant.model.model_id` (the base FP16 model) while the eval and
`weights_disk_mb` use the served compressed-tensors `checkpoint_path`; the bench must
target the same served weights. Second, `vllm bench serve` needs a running vLLM server that
nothing starts; the pod fix decides `bench serve` (start and drive a server) versus `bench
latency` (offline) and matches `parse_vllm_bench`'s expected JSON keys to whichever tool is
used. The two pure parsers and `NativeLlamaCppLatency` are not implicated. Until this gate
clears, no vLLM tok/s from this probe is trustworthy. The note lives on the class and module
docstrings in `latency/native.py`.

### 2026-07-13 Result store and figures are regenerated, not committed (SETTLED)
`results/` (the parquet + jsonl store, including the per-item predictions sidecars)
and `plots/` are git-ignored. They are throwaway output: the smoke store churns on
every local run, and a committed binary parquet is noise in the history. Reproducibility
is by deterministic re-run, not by checking the output in: each row carries the git SHA,
the resolved config hash, the model revision, and the seed, so a run reconstructs from
the config plus that provenance. This reverses the earlier "result rows are committed"
note in the `.gitignore` and README. The exception is deliberate: when the real pod runs
finish, the final headline figures and the exact result set behind the report are
committed under `reports/` as the citable record.

### 2026-07-13 Target the true 16GB A4000/L4, and cut the matrix to protect ~20h (SETTLED)
Chosen over a 4090-capped-at-16GB: the honest card is faithful to the stated
ceiling and cheaper per hour, at ~2-3x slower throughput. The full matrix would be
~35-50 GPU-h there, so the matrix is cut to fit. The cuts, each preserving the
contribution: drop GGUF Q8_0 (~FP16, uninteresting); OOD calibration axis on GPTQ +
AWQ only; PTQ sweep on a 2,000-item MMLU subset (full 14k only for the headline
rows fp16 / int4-gptq / qat-3bit-lora / distill-softlabel-topk); latency at batch
{1,4,16}, 20 trials, shared decode kernels measured once; the CoT reasoning arm on a
1,000-item slice and a 6-variant subset only; distillation training capped at 30M
tokens, 1 epoch, 0.5B student; online GKD replaced by a validation-only logit
comparison. GPU-hour totals stay provisional until WP3's latency rig measures real
throughput on the actual pod; the training-arm token budgets are re-confirmed
against that number before any GPU time is spent on WP6/WP7. Measure, do not assume.

### 2026-07-13 Reasoning arm: CoT-then-answer-token on a variant subset (SETTLED)
Keeps the "compression bites hardest on multi-step reasoning" claim. A single clean
answer token needs no-CoT, but reasoning needs CoT, so the reasoning arm lets the
model generate CoT and then reads the answer-letter logprob after an "Answer:"
trigger. Task: GPQA-Diamond (MMLU-Pro-with-CoT as the alternate). It needs
generation, so it is GPU-only (not CPU-smoke-friendly) and runs on a 1,000-item
slice and only the representative variants fp16, int4-gptq, int4-nf4,
ptq-3bit-torchao, qat-3bit-lora, distill-softlabel-topk. Config:
`configs/evals/reasoning-cot.yaml`.

### 2026-07-13 KD temperature axis recovered post-hoc, not retrained (SETTLED)
The adversarial cost check refuted the 18 GPU-hour estimate: a training-time KD
temperature sweep {1,2,4} is separate student training runs (caching only makes the
*teacher* pass free), ~15-24h honestly. Instead train one distilled student and
recover the temperature axis with post-hoc temperature scaling (a ~0-GPU
one-parameter fit, and the standard calibration baseline anyway). The sharper
question: does training-time soft-label KD beat a free post-hoc fix on ECE? The
online full-vocab GKD *training* run is dropped for a validation-only
offline-vs-online logit comparison on a few thousand tokens.

### 2026-07-13 Task instrument: three fixes, and the reasoning claim is kept (SETTLED)
The adversarial task check returned uncertain. MMLU answer-letter softmax is
workable but not clean, so three fixes are mandatory: (a) cyclic-permutation
(PriDe) scoring, because letter selection bias is itself compression-sensitive and
a delta-ECE across bit-widths could otherwise be a bias shift; (b) MMLU-Redux
labels for the final ECE (~6.5% label error, up to 57% in some subjects, hits ECE
harder than accuracy); (c) validate that the argmax letter agrees with the model's
free-generated answer on a slice. The reasoning claim is kept via the CoT arm above,
not via no-CoT MMLU-Pro (which would measure guessing calibration).

### 2026-07-13 Task: MMLU primary, ARC-Challenge secondary, GPQA CoT reasoning arm; GSM8K excluded (SETTLED)
Single answer-letter token gives the cleanest per-option probability (no
length-normalisation confound). ARC-Challenge (1,172 items, independent) is the
cheap secondary; the reasoning arm adds the difficulty axis the central hypothesis
needs. GSM8K stays out of the ECE analysis: its only usable confidence signal
(self-consistency vote share) is a different mechanism than the MCQ token
probability and costs K generations per item (optional accuracy-only check at most).

### 2026-07-13 Base model: Qwen2.5-3B-Instruct teacher, Qwen2.5-0.5B-Instruct student (SETTLED)
Cleanest same-family ladder (0.5/1.5/3/7B, one 151,936 tokenizer, identical dense
arch), deepest stock of official pre-quantised checkpoints (budget goes to
calibration, not baselines), Apache-2.0, non-reasoning so confidence stays clean.
Fallback Gemma-3-4B/1B (official int4 QAT reference, cross-vendor) remains a named
option for a later generalisation pass. Smoke model: SmolLM2-135M.

### 2026-07-13 Seeds: 1 for deterministic PTQ eval, 3 for training arms (SETTLED)
A PTQ eval on a fixed checkpoint is deterministic, so eval-set sampling noise is
quantified by the bootstrap CIs rather than by re-seeding. PTQ variants default to
1 seed. QAT and distillation carry training randomness and override to 3 seeds.

### 2026-07-13 QAT: one variant, 3-bit torchao QAT+LoRA on the teacher, full-param QAT on the 0.5B student (SETTLED, bit-width confirmed empirically)
Full-parameter QAT of a 3B does not fit 16GB (~36-48GB); only QAT+LoRA fits, so the
teacher QAT is LoRA-based. The 0.5B student is the one tier where genuine
full-parameter QAT fits (~4-8GB), which is the sharpest single calibration data
point. Bit-width 3: int4 PTQ on a 3B is essentially solved (QAT would look
pointless), 2-bit risks a null result for LoRA-QAT; 3-bit is where PTQ visibly
fails and QAT can claw it back. 3-bit is lightly battle-tested in torchao, so run
the free 3-bit PTQ arm and a CPU fake-quant smoke first; int4 (via Unsloth) is the
pre-decided fallback. Match the QAT arm to a torchao PTQ baseline with the
identical quant config so the only difference is training.

### 2026-07-13 Distillation route: offline top-k logit caching (SETTLED, verified)
Full-vocab soft targets are infeasible (~30TB for 100M tokens at 151,936 vocab).
Top-k (k=64) with a "rest" bucket (1 - sum(top-k), kept not renormalised, so
truncation does not bias ECE/Brier) costs ~38GB on disk and makes every student
run teacher-free. Online full-precision KD does not fit 16GB (needs a 4-bit teacher
+ logit chunking). Index dtype uint32 (vocab needs 18 bits). Do not use LoRA for
the calibration students: LoRA vs full-FT changes calibration and would confound
the central claim.
