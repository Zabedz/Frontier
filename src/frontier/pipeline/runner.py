"""The config-to-row orchestrator.

``run`` resolves a config, loads the backend once, and for each seed loads the eval
slice, scores it through the WP2 core, computes the WP1 calibration battery, stamps
provenance, and appends one ``ResultRow`` to the store. The provider factory and the
slice loader are injectable, so a CPU test drives the whole chain with a synthetic
provider and in-memory records: no model, no network.

``perplexity`` stays ``NaN`` (no held-out corpus yet). ``latency``, ``memory``, and
``tok_s_per_gb`` are filled once per run by the injectable ``latency_probe`` (the real
``default_latency`` rig by default); ``measure_latency=False`` restores the empty /
``NaN`` fields for a secondary profile joined to the primary run in analysis.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from frontier.backends.hf import HFLogitProvider, resolve_device
from frontier.eval.extract import exact_match, score_items, to_robustness, to_task_spec
from frontier.eval.loaders import load_arc_challenge, load_mmlu, load_mmlu_redux
from frontier.eval.provider import LogitProvider
from frontier.eval.records import EvalRecord
from frontier.io.predictions import PredictionRows, predictions_key, write_predictions_rows
from frontier.io.provenance import hardware_info, now_utc_iso, read_git_sha, stamp_provenance
from frontier.io.store import ResultStore, append_row
from frontier.latency.rig import LatencyMemory, default_latency
from frontier.metrics.bootstrap import accuracy_ci, ece_ci
from frontier.metrics.report import calibration_report, to_quality
from frontier.pipeline.config import DEFAULT_CONFIG_ROOT, ResolvedConfig, resolve_config
from frontier.schema import Backend, EvalSpec, ResultRow, RunMode, VariantConfig


class ProviderFactory(Protocol):
    """Builds the logit provider for a resolved variant on a device."""

    def __call__(self, variant: VariantConfig, device: str) -> LogitProvider: ...


class SliceLoader(Protocol):
    """Loads the eval slice for a spec and seed."""

    def __call__(self, spec: EvalSpec, *, seed: int) -> list[EvalRecord]: ...


class LatencyProbe(Protocol):
    """Measures the latency/memory record for a loaded provider on a device."""

    def __call__(
        self,
        provider: LogitProvider,
        resolved: ResolvedConfig,
        *,
        device: str,
        mode: RunMode,
    ) -> LatencyMemory: ...


def load_slice(spec: EvalSpec, *, seed: int) -> list[EvalRecord]:
    """Dispatch on ``task_name`` (and ``labels`` for MMLU) to the WP2 loaders.

    ``mmlu`` + ``labels="redux"`` uses the de-noised clean subset; ``mmlu`` +
    ``labels="raw"`` loads one ``cais/mmlu`` config (the smoke path). ``arc_challenge``
    has no redux equivalent. Any other task raises ``ValueError`` naming it (the GPQA
    reasoning arm arrives with the GPU generation WP).
    """
    if spec.task_name == "mmlu":
        if spec.labels == "redux":
            return load_mmlu_redux(policy="clean_subset", subset=spec.subset_size, seed=seed)
        return load_mmlu(spec.split, subset=spec.subset_size, seed=seed)
    if spec.task_name == "arc_challenge":
        return load_arc_challenge(spec.split, subset=spec.subset_size, seed=seed)
    raise ValueError(f"no eval loader for task {spec.task_name!r}")


def run(
    config_path: Path,
    *,
    eval_profile: str | None = None,
    mode: RunMode = "full",
    config_root: Path = DEFAULT_CONFIG_ROOT,
    results_root: Path = Path("results"),
    timestamp: str | None = None,
    provider_factory: ProviderFactory | None = None,
    slice_loader: SliceLoader | None = None,
    git_sha: str | None = None,
    latency_probe: LatencyProbe | None = None,
    measure_latency: bool = True,
    write_predictions: bool = True,
) -> list[ResultRow]:
    """Resolve config, score, compute metrics, assemble one row per seed, append.

    One row per seed in ``eval_spec.seeds`` (smoke has ``[0]`` -> one row). The provider
    is built once and reused across seeds, because the model load is the expensive step.
    Latency and memory are a property of the variant and the hardware, not of an eval
    seed, so they are measured once (before the seed loop) and shared across the rows;
    ``measure_latency=False`` (CLI ``--skip-latency``) leaves those fields empty for a
    secondary profile that will be joined to the primary run's latency in analysis.
    ``write_predictions=True`` (CLI default; ``--skip-predictions`` disables it) writes
    each seed's per-item ``(confidence, correct, gold, predicted)`` sidecar next to the
    appended row, the raw calibration signal the reliability figures read back.
    Returns the appended rows in seed order.
    """
    resolved = resolve_config(
        config_path, eval_profile=eval_profile, mode=mode, config_root=config_root
    )
    device = resolve_device(mode)
    provider = (
        provider_factory(resolved.variant, device)
        if provider_factory is not None
        else _default_provider(resolved, device)
    )
    load: Callable[..., list[EvalRecord]] = slice_loader or load_slice
    store = ResultStore(results_root)
    hardware = hardware_info(device=device)
    sha = git_sha if git_sha is not None else read_git_sha()

    if measure_latency:
        probe = latency_probe or default_latency
        lat_mem = probe(provider, resolved, device=device, mode=mode)
    else:
        lat_mem = LatencyMemory(latency=[], memory=[], tok_s_per_gb=math.nan)

    rows: list[ResultRow] = []
    for seed in resolved.eval_spec.seeds:
        records = load(resolved.eval_spec, seed=seed)
        out = score_items(
            records,
            provider,
            scheme=resolved.eval_spec.permutation_scheme,
            cot=resolved.eval_spec.cot,
        )
        correct = exact_match(out.predicted, out.gold)
        report = calibration_report(out.probs, out.gold)
        quality = to_quality(
            report,
            accuracy_ci=accuracy_ci(correct, rng=seed),
            ece_ci=ece_ci(out.confidence, correct, rng=seed),
            perplexity=math.nan,
            temperature=1.0,
            temperature_scaled=False,
        )
        prov = stamp_provenance(
            config_hash=resolved.config_hash,
            model_id=resolved.variant.model.model_id,
            model_revision=resolved.variant.model.model_revision,
            seed=seed,
            hardware=hardware,
            git_sha=sha,
            timestamp=timestamp or now_utc_iso(),
        )
        row = ResultRow(
            provenance=prov,
            backend=_build_backend(resolved, provider),
            variant_name=resolved.variant.name,
            family=resolved.variant.family,
            task=to_task_spec(resolved.eval_spec, num_items=len(records)),
            quality=quality,
            latency=lat_mem.latency,
            memory=lat_mem.memory,
            tok_s_per_gb=lat_mem.tok_s_per_gb,
            robustness=to_robustness(out.robustness),
        )
        append_row(row, store)
        if write_predictions:
            write_predictions_rows(
                PredictionRows(
                    confidence=out.confidence,
                    correct=correct,
                    gold=out.gold,
                    predicted=out.predicted,
                ),
                root=results_root,
                key=predictions_key(resolved.config_hash, seed, resolved.eval_spec.task_name),
            )
        rows.append(row)
    return rows


def _default_provider(resolved: ResolvedConfig, device: str) -> LogitProvider:
    return HFLogitProvider(
        model_id=resolved.variant.model.model_id,
        device=device,
        weight_dtype=str(resolved.backend["weight_dtype"]),
        revision=resolved.variant.model.model_revision,
    )


def _build_backend(resolved: ResolvedConfig, provider: LogitProvider) -> Backend:
    raw = resolved.backend
    return Backend(
        inference_backend=raw["inference_backend"],
        backend_version=getattr(provider, "backend_version", "unknown"),
        weight_dtype=raw["weight_dtype"],
        kv_cache_dtype=raw["kv_cache_dtype"],
        gpu_offload_layers=raw["gpu_offload_layers"],
        track=resolved.variant.track,
    )
