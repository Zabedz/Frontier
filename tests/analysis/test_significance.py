"""Pair resolution, the alignment guards, and the significance table.

A paired bootstrap over two variants scored on different items returns a plausible number
with no symptom, so every mismatch here has to raise or skip.
"""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from frontier.analysis.load import load_predictions_for_variant, load_tidy
from frontier.analysis.significance import (
    DEFAULT_REFERENCES_PATH,
    References,
    VariantPair,
    load_references,
    pair_significance,
    pair_significance_seeded,
    resolve_pairs,
    significance_table,
    to_frame,
)
from frontier.io.predictions import (
    LabelArray,
    PredictionRows,
    predictions_key,
    write_predictions_rows,
)
from frontier.io.store import ResultStore, append_row
from frontier.pipeline.cli import _excess_verdict
from frontier.schema import Family, ResultRow

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "io"))

from rows import sample_row

RESAMPLES = 99
N_ITEMS = 400
HIT_RATE = 0.65
HEADLINE_BINS = 7
INTERVAL_FIELDS = 3  # low, point, high per sweep entry
REFERENCES = {"hf": "fp16", "llama_cpp": "gguf-q8_0"}
QWEN_3B = "Qwen/Qwen2.5-3B-Instruct"
QWEN_05B = "Qwen/Qwen2.5-0.5B-Instruct"


def _gold(n: int = N_ITEMS) -> LabelArray:
    labels: LabelArray = np.random.default_rng(0).integers(0, 4, size=n).astype(np.intp)
    return labels


def _predictions(gold: LabelArray, *, seed: int, miscalibration: float = 0.0) -> PredictionRows:
    rng = np.random.default_rng(seed)
    confidence = np.clip(rng.uniform(0.3, 0.98, size=gold.shape[0]) + miscalibration, 0.0, 1.0)
    correct = rng.uniform(0.0, 1.0, size=gold.shape[0]) < HIT_RATE
    predicted = np.where(correct, gold, (gold + 1) % 4).astype(np.intp)
    return PredictionRows(
        confidence,
        correct.astype(np.bool_),
        gold,
        predicted,
        options=None,
        qid=None,
        order_probs=None,
    )


def _row(
    *,
    name: str,
    backend: str,
    track: str,
    config_hash: str,
    seed: int = 0,
    task_name: str = "mmlu",
    model: str | None = None,
    family: Family = "baseline",
) -> ResultRow:
    base = sample_row()
    return replace(
        base,
        variant_name=name,
        family=family,
        backend=replace(base.backend, inference_backend=backend, track=track),  # type: ignore[arg-type]
        provenance=replace(
            base.provenance,
            config_hash=config_hash,
            seed=seed,
            model_id=model or base.provenance.model_id,
        ),
        task=replace(base.task, task_name=task_name),
    )


def _store_with(
    tmp_path: Path, rows: list[ResultRow], *, sidecars: bool = True
) -> tuple[ResultStore, Path]:
    store = ResultStore(tmp_path)
    gold = _gold()
    for index, row in enumerate(rows):
        append_row(row, store)
        if sidecars:
            write_predictions_rows(
                _predictions(gold, seed=index, miscalibration=0.02 * index),
                root=tmp_path,
                key=predictions_key(
                    row.provenance.config_hash, row.provenance.seed, row.task.task_name
                ),
            )
    return store, tmp_path


def test_resolve_pairs_matches_each_variant_to_its_backend_reference(tmp_path: Path) -> None:
    store, _root = _store_with(
        tmp_path,
        [
            _row(name="fp16", backend="hf", track="A", config_hash="a" * 64),
            _row(name="int4-nf4", backend="hf", track="A", config_hash="b" * 64),
            _row(name="gguf-q8_0", backend="llama_cpp", track="B", config_hash="c" * 64),
            _row(name="gguf-q4_k_m", backend="llama_cpp", track="B", config_hash="d" * 64),
        ],
    )
    pairs, skipped = resolve_pairs(load_tidy(store), REFERENCES)
    assert {(pair.variant, pair.reference) for pair in pairs} == {
        ("int4-nf4", "fp16"),
        ("gguf-q4_k_m", "gguf-q8_0"),
    }
    assert {skip.variant for skip in skipped} == {"fp16", "gguf-q8_0"}
    assert all("is the reference" in skip.reason for skip in skipped)


def test_resolve_pairs_never_pairs_across_backends(tmp_path: Path) -> None:
    store, _root = _store_with(
        tmp_path,
        [
            _row(name="fp16", backend="hf", track="A", config_hash="a" * 64),
            _row(name="gguf-q4_k_m", backend="llama_cpp", track="B", config_hash="d" * 64),
        ],
    )
    pairs, skipped = resolve_pairs(load_tidy(store), REFERENCES)
    assert pairs == []
    assert any("gguf-q8_0 has no row" in skip.reason for skip in skipped)


def test_resolve_pairs_skips_a_backend_with_no_configured_reference(tmp_path: Path) -> None:
    store, _root = _store_with(
        tmp_path,
        [_row(name="ptq-3bit-torchao", backend="torchao", track="A", config_hash="e" * 64)],
    )
    pairs, skipped = resolve_pairs(load_tidy(store), REFERENCES)
    assert pairs == []
    assert skipped[0].reason == "no reference configured for backend torchao"


def test_resolve_pairs_pairs_a_multi_seed_variant_with_a_one_seed_reference(
    tmp_path: Path,
) -> None:
    store, _root = _store_with(
        tmp_path,
        [
            _row(name="fp16", backend="hf", track="A", config_hash="a" * 64, seed=0),
            _row(name="int4-nf4", backend="hf", track="A", config_hash="b" * 64, seed=0),
            _row(name="int4-nf4", backend="hf", track="A", config_hash="b" * 64, seed=1),
        ],
    )
    pairs, _skipped = resolve_pairs(load_tidy(store), REFERENCES)
    assert [pair.variant for pair in pairs] == ["int4-nf4"]


def test_resolve_pairs_skips_a_reference_scored_at_several_seeds(tmp_path: Path) -> None:
    store, _root = _store_with(
        tmp_path,
        [
            _row(name="fp16", backend="hf", track="A", config_hash="a" * 64, seed=0),
            _row(name="fp16", backend="hf", track="A", config_hash="a" * 64, seed=1),
            _row(name="int4-nf4", backend="hf", track="A", config_hash="b" * 64, seed=0),
        ],
    )
    pairs, skipped = resolve_pairs(load_tidy(store), REFERENCES)
    assert pairs == []
    assert any("one deterministic run" in skip.reason for skip in skipped)


def test_resolve_pairs_is_empty_on_an_empty_frame(tmp_path: Path) -> None:
    store = ResultStore(tmp_path)
    append_row(_row(name="fp16", backend="hf", track="A", config_hash="a" * 64), store)
    pairs, skipped = resolve_pairs(load_tidy(store, task_name="absent"), REFERENCES)
    assert pairs == []
    assert skipped == []


def _pair(variant: str = "int4-nf4", reference: str = "fp16") -> VariantPair:
    return VariantPair(variant=variant, reference=reference, task="mmlu", backend="hf", track="A")


def test_pair_significance_raises_when_the_sidecars_describe_different_items() -> None:
    gold = _gold()
    shifted = np.roll(gold, 1)
    with pytest.raises(ValueError, match="describe different items") as caught:
        pair_significance(
            _pair(),
            _predictions(gold, seed=1),
            _predictions(shifted, seed=2),
            n_resamples=RESAMPLES,
        )
    message = str(caught.value)
    assert "int4-nf4" in message
    assert "fp16" in message


def test_pair_significance_raises_when_the_sidecars_hold_different_item_counts() -> None:
    with pytest.raises(ValueError, match="sidecars hold"):
        pair_significance(
            _pair(),
            _predictions(_gold(N_ITEMS), seed=1),
            _predictions(_gold(N_ITEMS - 10), seed=2),
            n_resamples=RESAMPLES,
        )


def test_pair_significance_folds_the_headline_bin_count_into_the_sweep() -> None:
    gold = _gold()
    outcome = pair_significance(
        _pair(),
        _predictions(gold, seed=1),
        _predictions(gold, seed=2, miscalibration=0.05),
        n_bins=HEADLINE_BINS,
        sweep_bins=(5, 10),
        n_resamples=RESAMPLES,
    )
    assert set(outcome.delta_ece_sweep) == {5, HEADLINE_BINS, 10}
    assert outcome.delta_ece == outcome.delta_ece_sweep[HEADLINE_BINS]
    assert outcome.n_items == N_ITEMS
    assert outcome.n_bins == HEADLINE_BINS


def test_pair_significance_on_identical_inputs_reports_no_separation() -> None:
    gold = _gold()
    rows = _predictions(gold, seed=1)
    outcome = pair_significance(
        _pair(), rows, rows, n_bins=10, sweep_bins=(10,), n_resamples=RESAMPLES
    )
    assert outcome.delta_ece.point == 0.0
    assert outcome.damage_gap.point == 0.0
    assert not outcome.damage_gap.excludes_zero
    assert not outcome.damage_ratio.usable
    assert not outcome.delta_ece_sign_stable


def test_significance_table_skips_a_pair_whose_sidecar_is_missing(tmp_path: Path) -> None:
    store, root = _store_with(
        tmp_path,
        [
            _row(name="fp16", backend="hf", track="A", config_hash="a" * 64),
            _row(name="int4-nf4", backend="hf", track="A", config_hash="b" * 64),
        ],
        sidecars=False,
    )
    found, skipped = significance_table(
        load_tidy(store), root=root, references=REFERENCES, n_resamples=RESAMPLES
    )
    assert found == []
    assert any("no predictions sidecar" in skip.reason for skip in skipped)


def test_significance_table_runs_every_resolved_pair(tmp_path: Path) -> None:
    store, root = _store_with(
        tmp_path,
        [
            _row(name="fp16", backend="hf", track="A", config_hash="a" * 64),
            _row(name="int4-nf4", backend="hf", track="A", config_hash="b" * 64),
        ],
    )
    found, _skipped = significance_table(
        load_tidy(store),
        root=root,
        references=REFERENCES,
        sweep_bins=(5, 10),
        n_resamples=RESAMPLES,
    )
    assert len(found) == 1
    assert found[0].pair.variant == "int4-nf4"
    assert found[0].pair.reference == "fp16"


def test_to_frame_carries_the_sweep_as_json() -> None:
    gold = _gold()
    outcome = pair_significance(
        _pair(),
        _predictions(gold, seed=1),
        _predictions(gold, seed=2, miscalibration=0.05),
        n_bins=10,
        sweep_bins=(5, 10),
        n_resamples=RESAMPLES,
    )
    frame = to_frame([outcome])
    assert list(frame["variant"]) == ["int4-nf4"]
    sweep = json.loads(str(frame["delta_ece_sweep"].iloc[0]))
    assert set(sweep) == {"5", "10"}
    assert len(sweep["10"]) == INTERVAL_FIELDS


def test_to_frame_is_empty_for_no_results() -> None:
    assert to_frame([]).empty


def test_load_references_reads_the_shipped_config() -> None:
    references = load_references()
    assert references.default["hf"] == "fp16"
    assert references.default["llama_cpp"] == "gguf-q8_0"
    assert references.for_variant("torchao", QWEN_3B) == "fp16"
    assert references.for_variant("torchao", QWEN_05B) == "student-fp16"
    assert references.for_variant("hf", QWEN_05B) == "student-fp16"
    assert references.for_variant("vllm", QWEN_05B) == "fp16-vllm"
    assert DEFAULT_REFERENCES_PATH.exists()


def test_load_references_rejects_a_non_mapping_by_model_block(tmp_path: Path) -> None:
    path = tmp_path / "significance.yaml"
    path.write_text("references:\n  hf: fp16\nby_model:\n  - x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="'by_model' must be a mapping"):
        load_references(path)


def test_resolve_pairs_takes_a_per_model_reference(tmp_path: Path) -> None:
    store, _root = _store_with(
        tmp_path,
        [
            _row(name="fp16", backend="hf", track="A", config_hash="a" * 64, model=QWEN_3B),
            _row(name="ptq", backend="torchao", track="A", config_hash="b" * 64, model=QWEN_3B),
            _row(name="s-fp16", backend="hf", track="A", config_hash="c" * 64, model=QWEN_05B),
            _row(name="s-ptq", backend="torchao", track="A", config_hash="d" * 64, model=QWEN_05B),
        ],
    )
    references = References(
        default={"hf": "fp16", "torchao": "fp16"},
        by_model={QWEN_05B: {"hf": "s-fp16", "torchao": "s-fp16"}},
    )
    pairs, _skipped = resolve_pairs(load_tidy(store), references)
    assert {(pair.variant, pair.reference) for pair in pairs} == {
        ("ptq", "fp16"),
        ("s-ptq", "s-fp16"),
    }


def test_resolve_pairs_refuses_a_reference_scoring_another_model(tmp_path: Path) -> None:
    store, _root = _store_with(
        tmp_path,
        [
            _row(name="fp16", backend="hf", track="A", config_hash="a" * 64, model=QWEN_3B),
            _row(name="s-ptq", backend="hf", track="A", config_hash="d" * 64, model=QWEN_05B),
        ],
    )
    pairs, skipped = resolve_pairs(load_tidy(store), {"hf": "fp16"})
    assert pairs == []
    assert any(
        skip.variant == "s-ptq" and f"scores {QWEN_3B}, not {QWEN_05B}" in skip.reason
        for skip in skipped
    )


def test_load_references_rejects_a_file_with_no_references_mapping(tmp_path: Path) -> None:
    path = tmp_path / "significance.yaml"
    path.write_text("something_else: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no top-level 'references' mapping"):
        load_references(path)


def test_load_references_rejects_a_non_mapping_references_block(tmp_path: Path) -> None:
    path = tmp_path / "significance.yaml"
    path.write_text("references:\n  - fp16\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a mapping"):
        load_references(path)


def test_load_references_rejects_a_non_string_reference(tmp_path: Path) -> None:
    path = tmp_path / "significance.yaml"
    path.write_text("references:\n  hf: 17\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must be a variant name"):
        load_references(path)


def _worse_than(rows: PredictionRows, *, bump: float = 0.25, flips: int = 8) -> PredictionRows:
    """The same items, made much more overconfident and slightly less accurate.

    The bump is large because the base fixture draws confidence independently of
    correctness, so its ECE already sits near 0.17.
    """
    correct = rows.correct.copy()
    correct[np.flatnonzero(rows.correct)[:flips]] = False
    return PredictionRows(
        np.clip(rows.confidence + bump, 0.0, 1.0),
        correct,
        rows.gold,
        rows.predicted,
        options=None,
        qid=None,
        order_probs=None,
    )


def test_pair_significance_points_from_the_reference_to_the_variant() -> None:
    gold = _gold()
    reference = _predictions(gold, seed=1)
    outcome = pair_significance(
        _pair(),
        reference,
        _worse_than(reference),
        n_bins=10,
        sweep_bins=(10,),
        n_resamples=RESAMPLES,
    )
    assert outcome.delta_accuracy.point < 0.0
    assert outcome.delta_confidence.point > 0.0
    assert outcome.delta_ece.point > 0.0
    assert outcome.damage_gap.point > 0.0
    assert outcome.damage_ratio.point > 1.0


def test_pair_significance_flips_every_sign_when_the_arguments_are_swapped() -> None:
    gold = _gold()
    reference = _predictions(gold, seed=1)
    worse = _worse_than(reference)
    mirrored = pair_significance(
        _pair(), worse, reference, n_bins=10, sweep_bins=(10,), n_resamples=RESAMPLES
    )
    assert mirrored.delta_accuracy.point > 0.0
    assert mirrored.delta_confidence.point < 0.0
    assert mirrored.delta_ece.point < 0.0
    assert mirrored.damage_gap.point < 0.0


def test_delta_ece_sweep_all_exclude_zero_tracks_the_weakest_bin_count() -> None:
    gold = _gold()
    reference = _predictions(gold, seed=1)
    separated = pair_significance(
        _pair(),
        reference,
        _worse_than(reference),
        n_bins=10,
        sweep_bins=(5, 10, 20),
        n_resamples=RESAMPLES,
    )
    assert separated.delta_ece_sweep_all_exclude_zero
    assert separated.delta_ece_sign_stable
    identical = pair_significance(
        _pair(),
        reference,
        reference,
        n_bins=10,
        sweep_bins=(5, 10, 20),
        n_resamples=RESAMPLES,
    )
    assert not identical.delta_ece_sweep_all_exclude_zero


def test_significance_table_refuses_a_variant_spanning_two_config_hashes(tmp_path: Path) -> None:
    """Pooling a re-run's second row would double n with the seed and gold guards satisfied."""
    store, root = _store_with(
        tmp_path,
        [
            _row(name="fp16", backend="hf", track="A", config_hash="a" * 64),
            _row(name="int4-nf4", backend="hf", track="A", config_hash="b" * 64),
            _row(name="int4-nf4", backend="hf", track="A", config_hash="f" * 64),
        ],
    )
    found, skipped = significance_table(
        load_tidy(store), root=root, references=REFERENCES, n_resamples=RESAMPLES
    )
    assert found == []
    assert any("spans 2 config hashes" in skip.reason for skip in skipped)


def test_load_predictions_for_variant_refuses_two_config_hashes(tmp_path: Path) -> None:
    store, root = _store_with(
        tmp_path,
        [
            _row(name="int4-nf4", backend="hf", track="A", config_hash="b" * 64),
            _row(name="int4-nf4", backend="hf", track="A", config_hash="f" * 64),
        ],
    )
    with pytest.raises(ValueError, match="spans 2 config hashes"):
        load_predictions_for_variant(
            load_tidy(store), variant_name="int4-nf4", task_name="mmlu", root=root
        )


def test_to_frame_column_contract_is_stable() -> None:
    gold = _gold()
    reference = _predictions(gold, seed=1)
    outcome = pair_significance(
        _pair(),
        reference,
        _worse_than(reference),
        n_bins=10,
        sweep_bins=(5, 10),
        n_resamples=RESAMPLES,
    )
    frame = to_frame([outcome])
    assert set(frame.columns) == {
        "variant",
        "reference",
        "task",
        "backend",
        "track",
        "crosses_model",
        "n_items",
        "n_bins",
        "delta_accuracy",
        "delta_accuracy_low",
        "delta_accuracy_high",
        "delta_confidence",
        "delta_confidence_low",
        "delta_confidence_high",
        "confidence_shifted",
        "delta_ece",
        "delta_ece_low",
        "delta_ece_high",
        "excess_damage",
        "excess_damage_low",
        "excess_damage_high",
        "calibration_damaged",
        "excess_withheld",
        "reference_gap",
        "variant_gap",
        "variant_gaps",
        "delta_auroc",
        "delta_auroc_low",
        "delta_auroc_high",
        "discrimination_moved",
        "damage_gap",
        "damage_gap_low",
        "damage_gap_high",
        "damage_ratio",
        "damage_ratio_low",
        "damage_ratio_high",
        "damage_ratio_usable",
        "damage_ratio_nonfinite_resamples",
        "accuracy_damage",
        "accuracy_damage_low",
        "accuracy_damage_high",
        "delta_ece_sign_stable",
        "delta_ece_sweep",
        "n_seeds",
        "seed_spread",
        "across_seeds",
    }
    low, point, high = json.loads(str(frame["delta_ece_sweep"].iloc[0]))["10"]
    assert (low, point, high) == (
        outcome.delta_ece_sweep[10].low,
        outcome.delta_ece_sweep[10].point,
        outcome.delta_ece_sweep[10].high,
    )
    assert low <= point <= high


def _less_accurate_only(rows: PredictionRows, *, flips: int = 120) -> PredictionRows:
    """The same stated confidence, more answers wrong. The confidence-invariant null."""
    correct = rows.correct.copy()
    correct[np.flatnonzero(rows.correct)[:flips]] = False
    return PredictionRows(
        rows.confidence.copy(),
        correct,
        rows.gold,
        rows.predicted,
        options=None,
        qid=None,
        order_probs=None,
    )


def test_confidence_is_flat_when_only_accuracy_moved() -> None:
    """The control the damage ratio needs.

    An overconfident model that only loses accuracy still shows an ECE delta and a damage
    gap, because ECE tracks mean confidence minus accuracy. Nothing was said about the
    model's confidence, and this is the statistic that reports that.
    """
    gold = _gold()
    reference = _predictions(gold, seed=1)
    outcome = pair_significance(
        _pair(),
        reference,
        _less_accurate_only(reference),
        n_bins=10,
        sweep_bins=(10,),
        n_resamples=RESAMPLES,
    )
    assert outcome.delta_accuracy.point < 0.0
    assert outcome.delta_confidence.point == 0.0
    assert not outcome.confidence_shifted
    assert not outcome.delta_confidence.excludes_zero


def test_confidence_shifted_reports_a_real_overconfidence_move() -> None:
    gold = _gold()
    reference = _predictions(gold, seed=1)
    outcome = pair_significance(
        _pair(),
        reference,
        _worse_than(reference),
        n_bins=10,
        sweep_bins=(10,),
        n_resamples=RESAMPLES,
    )
    assert outcome.confidence_shifted
    assert outcome.delta_confidence.excludes_zero


def test_seeded_pair_at_one_seed_equals_the_one_seed_path() -> None:
    gold = _gold()
    reference = _predictions(gold, seed=1)
    variant = _predictions(gold, seed=2, miscalibration=0.1)
    one = pair_significance(_pair(), reference, variant, n_resamples=RESAMPLES)
    seeded = pair_significance_seeded(_pair(), reference, [variant], n_resamples=RESAMPLES)
    for name in ("delta_accuracy", "delta_confidence", "delta_ece", "damage_gap"):
        assert getattr(seeded, name) == getattr(one, name), name
    assert seeded.delta_ece_sweep == one.delta_ece_sweep
    assert seeded.damage_ratio.point == one.damage_ratio.point
    assert seeded.damage_ratio.denominator == one.damage_ratio.denominator


def test_seeded_pair_reports_the_seed_count_and_each_quantity_spread() -> None:
    gold = _gold()
    reference = _predictions(gold, seed=1)
    seeds = [_predictions(gold, seed=seed, miscalibration=0.1) for seed in (2, 3, 4)]
    outcome = pair_significance_seeded(_pair(), reference, seeds, n_resamples=RESAMPLES)
    assert outcome.n_seeds == len(seeds)
    low, high = outcome.seed_spread["delta_ece"]
    assert low <= outcome.delta_ece.point <= high
    assert set(outcome.seed_spread) == {
        "delta_accuracy",
        "delta_confidence",
        "delta_ece",
        "excess_damage",
        "delta_auroc",
        "damage_gap",
    }
    assert set(outcome.across_seeds) == {
        "delta_accuracy",
        "delta_confidence",
        "delta_ece",
        "excess_damage",
        "delta_auroc",
    }
    assert outcome.verdict_interval("excess_damage") is outcome.across_seeds["excess_damage"]


def test_one_seed_verdicts_read_the_question_only_intervals() -> None:
    gold = _gold()
    reference = _predictions(gold, seed=1)
    outcome = pair_significance(_pair(), reference, _worse_than(reference), n_resamples=RESAMPLES)
    assert outcome.across_seeds == {}
    assert outcome.verdict_interval("excess_damage") is outcome.excess_damage
    assert outcome.verdict_interval("delta_auroc") is outcome.delta_auroc


def test_seeded_pair_refuses_a_seed_copy_over_other_items() -> None:
    gold = _gold()
    with pytest.raises(ValueError, match="describe different items"):
        pair_significance_seeded(
            _pair(),
            _predictions(gold, seed=1),
            [_predictions(gold, seed=2), _predictions(np.roll(gold, 1), seed=3)],
            n_resamples=RESAMPLES,
        )


DISTILLED = "distill-softlabel-topk"
IS_REFERENCE = "is the reference for its backend"
# (variant, family, backend, track, model, seeds) as banked in each store on the Mac.
BANKED_STORES: dict[str, list[tuple[str, Family, str, str, str, tuple[int, ...]]]] = {
    "results": [
        ("fp16", "baseline", "hf", "A", QWEN_3B, (0,)),
        ("int4-nf4", "ptq", "hf", "A", QWEN_3B, (0,)),
        ("int8-weightonly", "ptq", "hf", "A", QWEN_3B, (0,)),
        ("gguf-q8_0", "ptq", "llama_cpp", "B", QWEN_3B, (0,)),
        ("gguf-q5_k_m", "ptq", "llama_cpp", "B", QWEN_3B, (0,)),
        ("gguf-q4_k_m", "ptq", "llama_cpp", "B", QWEN_3B, (0,)),
        ("int4-gptq", "ptq", "vllm", "B", QWEN_3B, (0,)),
        ("int4-awq", "ptq", "vllm", "B", QWEN_3B, (0,)),
        ("int8-w8a8", "ptq", "vllm", "B", QWEN_3B, (0,)),
        ("fp16-vllm", "baseline", "vllm", "B", QWEN_3B, (0,)),
        ("int4-gptq-ood", "ptq", "vllm", "B", QWEN_3B, (0,)),
        ("int4-awq-ood", "ptq", "vllm", "B", QWEN_3B, (0,)),
    ],
    "results-qat": [
        ("student-fp16", "baseline", "hf", "A", QWEN_05B, (0,)),
        ("student-ptq-3bit-torchao", "ptq", "torchao", "A", QWEN_05B, (0,)),
        ("fp16", "baseline", "hf", "A", QWEN_3B, (0,)),
        ("ptq-3bit-torchao", "ptq", "torchao", "A", QWEN_3B, (0,)),
        ("student-qat-3bit-full", "qat", "torchao", "A", QWEN_05B, (0, 1, 2)),
        ("qat-3bit-lora", "qat", "torchao", "A", QWEN_3B, (0, 1, 2)),
    ],
    "results-seeds": [
        ("fp16-vllm", "baseline", "vllm", "B", QWEN_3B, (0,)),
        ("int4-gptq", "ptq", "vllm", "B", QWEN_3B, (0, 1, 2)),
        ("int4-awq", "ptq", "vllm", "B", QWEN_3B, (0, 1, 2)),
    ],
}
BANKED_PAIRS = {
    "results": {
        ("int4-nf4", "fp16"),
        ("int8-weightonly", "fp16"),
        ("gguf-q5_k_m", "gguf-q8_0"),
        ("gguf-q4_k_m", "gguf-q8_0"),
        ("int4-gptq", "fp16-vllm"),
        ("int4-awq", "fp16-vllm"),
        ("int8-w8a8", "fp16-vllm"),
        ("int4-gptq-ood", "fp16-vllm"),
        ("int4-awq-ood", "fp16-vllm"),
    },
    "results-qat": {
        ("student-ptq-3bit-torchao", "student-fp16"),
        ("ptq-3bit-torchao", "fp16"),
        ("student-qat-3bit-full", "student-fp16"),
        ("qat-3bit-lora", "fp16"),
    },
    "results-seeds": {("int4-gptq", "fp16-vllm"), ("int4-awq", "fp16-vllm")},
}
BANKED_SKIPS = {
    "results": {"fp16", "fp16-vllm", "gguf-q8_0"},
    "results-qat": {"student-fp16", "fp16"},
    "results-seeds": {"fp16-vllm"},
}
# The baseline family pair: the off-the-shelf 0.5B against the teacher.
FAMILY_PAIRS = {"results-qat": {("student-fp16", "fp16")}}


def _banked_rows(store: str) -> list[ResultRow]:
    return [
        _row(
            name=name,
            backend=backend,
            track=track,
            config_hash=f"{index:064x}",
            seed=seed,
            model=model,
            family=family,
        )
        for index, (name, family, backend, track, model, seeds) in enumerate(BANKED_STORES[store])
        for seed in seeds
    ]


@pytest.mark.parametrize("store", sorted(BANKED_STORES))
def test_the_shipped_references_keep_every_banked_pair(tmp_path: Path, store: str) -> None:
    banked, _root = _store_with(tmp_path, _banked_rows(store), sidecars=False)
    pairs, skipped = resolve_pairs(load_tidy(banked), load_references())
    same_model = {(pair.variant, pair.reference) for pair in pairs if not pair.crosses_model}
    crossing = {(pair.variant, pair.reference) for pair in pairs if pair.crosses_model}
    assert same_model == BANKED_PAIRS[store]
    assert crossing == FAMILY_PAIRS.get(store, set())
    assert {(skip.variant, skip.reason) for skip in skipped} == {
        (variant, IS_REFERENCE) for variant in BANKED_SKIPS[store]
    }


def _distill_rows(
    *, with_teacher: bool = True, teacher_seeds: tuple[int, ...] = (0,)
) -> list[ResultRow]:
    rows = [
        _row(name="student-fp16", backend="hf", track="A", config_hash="b" * 64, model=QWEN_05B)
    ]
    rows += [
        _row(
            name=DISTILLED,
            backend="hf",
            track="A",
            config_hash="c" * 64,
            seed=seed,
            model=QWEN_05B,
            family="distill",
        )
        for seed in (0, 1, 2)
    ]
    if with_teacher:
        rows += [
            _row(
                name="fp16", backend="hf", track="A", config_hash="a" * 64, seed=seed, model=QWEN_3B
            )
            for seed in teacher_seeds
        ]
    return rows


def test_a_distilled_student_pairs_with_its_own_model_and_with_the_teacher(
    tmp_path: Path,
) -> None:
    store, _root = _store_with(tmp_path, _distill_rows(), sidecars=False)
    pairs, _skipped = resolve_pairs(load_tidy(store), load_references())
    assert {(pair.variant, pair.reference, pair.crosses_model) for pair in pairs} == {
        (DISTILLED, "student-fp16", False),
        (DISTILLED, "fp16", True),
        ("student-fp16", "fp16", True),
    }


def test_the_family_pair_is_skipped_without_the_teacher_row(tmp_path: Path) -> None:
    store, _root = _store_with(tmp_path, _distill_rows(with_teacher=False), sidecars=False)
    pairs, skipped = resolve_pairs(load_tidy(store), load_references())
    assert [(pair.variant, pair.reference) for pair in pairs] == [(DISTILLED, "student-fp16")]
    assert (DISTILLED, "family reference fp16 has no row on task mmlu") in {
        (skip.variant, skip.reason) for skip in skipped
    }


def test_a_family_reference_at_several_seeds_is_skipped(tmp_path: Path) -> None:
    store, _root = _store_with(tmp_path, _distill_rows(teacher_seeds=(0, 1)), sidecars=False)
    _pairs, skipped = resolve_pairs(load_tidy(store), load_references())
    assert any(
        skip.variant == DISTILLED and skip.reason.startswith("family reference fp16 is scored")
        for skip in skipped
    )


def test_references_without_family_extras_pair_as_before(tmp_path: Path) -> None:
    store, _root = _store_with(tmp_path, _distill_rows(), sidecars=False)
    references = References(default={"hf": "fp16"}, by_model={QWEN_05B: {"hf": "student-fp16"}})
    pairs, _skipped = resolve_pairs(load_tidy(store), references)
    assert [(pair.variant, pair.reference, pair.crosses_model) for pair in pairs] == [
        (DISTILLED, "student-fp16", False)
    ]


def test_load_references_reads_the_shipped_family_extras() -> None:
    references = load_references()
    assert references.family_reference("distill", "hf") == "fp16"
    assert references.family_reference("baseline", "hf") == "fp16"
    assert references.family_reference("distill", "vllm") is None
    assert references.family_reference("ptq", "hf") is None


def test_load_references_rejects_a_non_mapping_family_block(tmp_path: Path) -> None:
    path = tmp_path / "significance.yaml"
    path.write_text("references:\n  hf: fp16\nalso_by_family:\n  - x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="'also_by_family' must be a mapping"):
        load_references(path)


def test_significance_table_runs_both_distilled_pairs_end_to_end(tmp_path: Path) -> None:
    store, root = _store_with(tmp_path, _distill_rows())
    found, _skipped = significance_table(
        load_tidy(store), root=root, references=load_references(), n_resamples=RESAMPLES
    )
    frame = to_frame(found)
    distilled = frame[frame["variant"] == DISTILLED]
    assert dict(zip(distilled["reference"], distilled["crosses_model"], strict=True)) == {
        "student-fp16": False,
        "fp16": True,
    }
    assert set(distilled["n_seeds"]) == {3}


def test_the_family_pair_stands_when_the_per_model_reference_scores_another_model(
    tmp_path: Path,
) -> None:
    """With no per-model entry the default reference is the teacher's row, refused for the
    per-model pair; the family pair still measures the student against it."""
    store, _root = _store_with(tmp_path, _distill_rows(), sidecars=False)
    references = References(default={"hf": "fp16"}, also_by_family={"distill": {"hf": "fp16"}})
    pairs, skipped = resolve_pairs(load_tidy(store), references)
    assert {(pair.variant, pair.reference, pair.crosses_model) for pair in pairs} == {
        (DISTILLED, "fp16", True)
    }
    assert (DISTILLED, f"reference fp16 scores {QWEN_3B}, not {QWEN_05B}") in {
        (skip.variant, skip.reason) for skip in skipped
    }


OVER = 0.25  # pushes the fixture's mean confidence well above its hit rate
UNDER = -0.3  # and well below it


def test_excess_is_withheld_when_the_variant_errs_the_other_way() -> None:
    gold = _gold()
    reference = _predictions(gold, seed=1, miscalibration=OVER)
    variant = _predictions(gold, seed=2, miscalibration=UNDER)
    outcome = pair_significance(_pair(), reference, variant, n_resamples=RESAMPLES)
    assert outcome.reference_gap > 0.0 > outcome.variant_gap
    assert outcome.excess_withheld
    assert not outcome.calibration_damaged
    frame = to_frame([outcome])
    assert bool(frame["excess_withheld"].iloc[0])
    assert frame["calibration_damaged"].iloc[0] is None
    assert json.loads(frame["variant_gaps"].iloc[0]) == [outcome.variant_gap]


def test_excess_stands_when_both_sides_err_the_same_way() -> None:
    gold = _gold()
    reference = _predictions(gold, seed=1, miscalibration=OVER)
    outcome = pair_significance(_pair(), reference, _worse_than(reference), n_resamples=RESAMPLES)
    assert outcome.reference_gap > 0.0
    assert outcome.variant_gap > 0.0
    assert not outcome.excess_withheld
    assert outcome.calibration_damaged


def test_one_seed_erring_the_other_way_withholds_the_seeded_excess() -> None:
    gold = _gold()
    reference = _predictions(gold, seed=1, miscalibration=OVER)
    seeds = [
        _predictions(gold, seed=2, miscalibration=OVER),
        _predictions(gold, seed=3, miscalibration=OVER),
        _predictions(gold, seed=4, miscalibration=UNDER),
    ]
    outcome = pair_significance_seeded(_pair(), reference, seeds, n_resamples=RESAMPLES)
    assert len(outcome.variant_gaps) == len(seeds)
    assert outcome.variant_gap > 0.0
    assert outcome.excess_withheld


def test_the_cli_names_a_withheld_excess_and_both_gaps() -> None:
    gold = _gold()
    reference = _predictions(gold, seed=1, miscalibration=OVER)
    variant = _predictions(gold, seed=2, miscalibration=UNDER)
    outcome = pair_significance(_pair(), reference, variant, n_resamples=RESAMPLES)
    text = _excess_verdict(outcome)
    assert text.startswith("excess damage withheld")
    assert f"{outcome.reference_gap:+.4f}" in text
    assert f"{outcome.variant_gap:+.4f}" in text
