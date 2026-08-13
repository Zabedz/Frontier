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
    VariantPair,
    load_references,
    pair_significance,
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
from frontier.schema import ResultRow

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "io"))

from rows import sample_row

RESAMPLES = 99
N_ITEMS = 400
HIT_RATE = 0.65
HEADLINE_BINS = 7
INTERVAL_FIELDS = 3  # low, point, high per sweep entry
REFERENCES = {"hf": "fp16", "llama_cpp": "gguf-q8_0"}


def _gold(n: int = N_ITEMS) -> LabelArray:
    labels: LabelArray = np.random.default_rng(0).integers(0, 4, size=n).astype(np.intp)
    return labels


def _predictions(gold: LabelArray, *, seed: int, miscalibration: float = 0.0) -> PredictionRows:
    rng = np.random.default_rng(seed)
    confidence = np.clip(rng.uniform(0.3, 0.98, size=gold.shape[0]) + miscalibration, 0.0, 1.0)
    correct = rng.uniform(0.0, 1.0, size=gold.shape[0]) < HIT_RATE
    predicted = np.where(correct, gold, (gold + 1) % 4).astype(np.intp)
    return PredictionRows(confidence, correct.astype(np.bool_), gold, predicted)


def _row(
    *,
    name: str,
    backend: str,
    track: str,
    config_hash: str,
    seed: int = 0,
    task_name: str = "mmlu",
) -> ResultRow:
    base = sample_row()
    return replace(
        base,
        variant_name=name,
        backend=replace(base.backend, inference_backend=backend, track=track),  # type: ignore[arg-type]
        provenance=replace(base.provenance, config_hash=config_hash, seed=seed),
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


def test_resolve_pairs_skips_when_the_seed_sets_differ(tmp_path: Path) -> None:
    store, _root = _store_with(
        tmp_path,
        [
            _row(name="fp16", backend="hf", track="A", config_hash="a" * 64, seed=0),
            _row(name="int4-nf4", backend="hf", track="A", config_hash="b" * 64, seed=0),
            _row(name="int4-nf4", backend="hf", track="A", config_hash="b" * 64, seed=1),
        ],
    )
    pairs, skipped = resolve_pairs(load_tidy(store), REFERENCES)
    assert pairs == []
    assert any("seed sets differ" in skip.reason for skip in skipped)


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
    assert references["hf"] == "fp16"
    assert references["llama_cpp"] == "gguf-q8_0"
    assert DEFAULT_REFERENCES_PATH.exists()


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
        np.clip(rows.confidence + bump, 0.0, 1.0), correct, rows.gold, rows.predicted
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
        "n_items",
        "n_bins",
        "delta_accuracy",
        "delta_accuracy_low",
        "delta_accuracy_high",
        "delta_ece",
        "delta_ece_low",
        "delta_ece_high",
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
    }
    low, point, high = json.loads(str(frame["delta_ece_sweep"].iloc[0]))["10"]
    assert (low, point, high) == (
        outcome.delta_ece_sweep[10].low,
        outcome.delta_ece_sweep[10].point,
        outcome.delta_ece_sweep[10].high,
    )
    assert low <= point <= high
