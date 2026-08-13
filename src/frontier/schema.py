"""Typed contracts for the benchmark: the variant config and the append-only result row.

Serialisation, validation, and the runner that populates these are written against this
contract and live elsewhere.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Family = Literal["baseline", "ptq", "qat", "distill"]
Track = Literal["A", "B"]
CalibrationCorpus = Literal["none", "in_domain", "ood"]
PromptStyle = Literal["zeroshot", "fiveshot"]
Scoring = Literal["letter_softmax", "acc_norm"]
PermutationScheme = Literal["none", "cyclic"]
Labels = Literal["raw", "redux"]
RunMode = Literal["smoke", "full"]


# The result row. Required records come first so they cannot be omitted.
@dataclass(frozen=True, slots=True)
class Provenance:
    """Everything needed to reproduce a row."""

    git_sha: str
    config_hash: str
    model_id: str
    model_revision: str
    hardware_id: str
    driver_version: str
    cuda_version: str
    seed: int
    timestamp: str


@dataclass(frozen=True, slots=True)
class Backend:
    """The inference backend, a first-class field on every measurement.

    An INT4 number and an FP16 number share a latency column only if they share a backend.
    ``track`` is the A/B split described in ``docs/architecture.md``.
    """

    inference_backend: Literal["hf", "vllm", "llama_cpp", "torchao"]
    backend_version: str
    weight_dtype: str
    kv_cache_dtype: str
    gpu_offload_layers: int
    track: Track


@dataclass(frozen=True, slots=True)
class TaskSpec:
    task_name: str
    split: str
    num_items: int
    prompt_style: PromptStyle
    scoring: Scoring
    permutation_scheme: PermutationScheme
    labels: Labels
    cot: bool


@dataclass(frozen=True, slots=True)
class Quality:
    """Task score plus the calibration battery.

    One ECE is never enough: ``ece_bin_sweep`` maps bin count to ECE, ``ece_equal_mass_ace``
    is the lower-bias companion, and the headline anchors on the bin-free
    ``brier_reliability`` term.
    """

    accuracy: float
    accuracy_ci_low: float
    accuracy_ci_high: float
    ece_equal_width: float
    ece_equal_mass_ace: float
    ece_bin_sweep: dict[int, float]
    ece_ci_low: float
    ece_ci_high: float
    brier: float
    brier_reliability: float
    brier_resolution: float
    brier_uncertainty: float
    nll: float
    perplexity: float
    temperature_scaled: bool
    temperature: float


@dataclass(frozen=True, slots=True)
class Robustness:
    """Permutation sensitivity of the calibration instrument (methodology section 2).

    The letter-selection confound moves with compression, so its magnitude is reported as
    its own block beside the calibration battery.
    """

    permutation_consistency: float  # mean fraction of cyclic orders already at the debiased answer
    letter_bias: float  # distance of the estimated per-letter-position prior from uniform
    debias_flip_rate: float  # fraction of items whose answer the debiasing changed


@dataclass(frozen=True, slots=True)
class MachineState:
    """GPU clock and thermal state at measurement time.

    Clocks usually cannot be locked on RunPod, so ``clocks_locked`` records the reality and
    the logged clock range is what defends the latency number.
    """

    gpu_clock_sm_mhz: int
    gpu_clock_mem_mhz: int
    gpu_temp_c: int
    power_w: float
    clocks_locked: bool
    clock_drift_flag: bool


@dataclass(frozen=True, slots=True)
class Latency:
    """One entry per batch size. TTFT and inter-token latency stay separate."""

    batch_size: int
    ttft_median_ms: float
    ttft_p95_ms: float
    itl_median_ms: float
    itl_p95_ms: float
    throughput_tok_s: float
    n_trials: int
    warmup_discarded: int
    machine_state: MachineState


@dataclass(frozen=True, slots=True)
class Memory:
    """One entry per (batch size, context length)."""

    batch_size: int
    context_len: int
    peak_vram_mb: float
    weights_disk_mb: float
    weights_resident_mb: float
    kv_cache_mb: float


@dataclass(frozen=True, slots=True)
class ResultRow:
    """One variant's full measurement.

    ``provenance`` and ``backend`` have no defaults, so a row whose kernel or git SHA went
    unrecorded does not construct.
    """

    provenance: Provenance
    backend: Backend
    variant_name: str
    family: Family
    task: TaskSpec
    quality: Quality
    latency: list[Latency]
    memory: list[Memory]
    tok_s_per_gb: float
    robustness: Robustness | None = None  # None where the eval used permutation_scheme="none"


# The variant config: one YAML file fully defines a variant.
@dataclass(frozen=True, slots=True)
class ModelSpec:
    model_id: str
    model_revision: str = "main"
    student_id: str | None = None
    student_revision: str = "main"


@dataclass(frozen=True, slots=True)
class QuantSpec:
    """One variant's quantisation settings.

    ``calibration_seed`` picks the shuffle that selects ``calibration_samples`` rows from the
    corpus, so it is an input to the produced weights and belongs in the config hash
    alongside the corpus. It stays ``None`` for a data-free method (bitsandbytes NF4, the
    GGUF k-quants), and a variant that names a corpus while omitting it is rejected by the
    producer.
    """

    method: str
    bit_width: int
    group_size: int
    calibration_corpus: CalibrationCorpus = "none"
    calibration_samples: int = 0
    calibration_seed: int | None = None


@dataclass(frozen=True, slots=True)
class DistillSpec:
    mode: Literal["hard_label", "soft_label_topk"]
    top_k: int = 64
    train_tokens: int = 50_000_000
    epochs: int = 1
    kd_temperature: float = 1.0


@dataclass(frozen=True, slots=True)
class EvalSpec:
    task_name: str
    split: str = "test"
    subset_size: int | None = None
    prompt_style: PromptStyle = "zeroshot"
    scoring: Scoring = "letter_softmax"
    permutation_scheme: PermutationScheme = "cyclic"
    labels: Labels = "redux"
    cot: bool = False
    seeds: tuple[int, ...] = (0, 1, 2)


@dataclass(frozen=True, slots=True)
class LatencySpec:
    batch_sizes: tuple[int, ...] = (1, 4, 16)
    context_lengths: tuple[int, ...] = (512, 2048)
    n_trials: int = 20
    warmup: int = 5


@dataclass(frozen=True, slots=True)
class VariantConfig:
    name: str
    family: Family
    track: Track
    model: ModelSpec
    eval: EvalSpec
    latency: LatencySpec = field(default_factory=LatencySpec)
    quant: QuantSpec | None = None
    distill: DistillSpec | None = None
