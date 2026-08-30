"""The repairability table: the fit-and-report chain, the leak guard, and the skip policy."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from frontier.analysis.holdout import NotRecalibratableError, is_fit
from frontier.analysis.load import load_split_predictions, load_tidy
from frontier.analysis.repairability import (
    REPORT_FLOOR,
    Repairability,
    RepairabilityPair,
    check_alignment,
    fingerprint,
    pairs_to_frame,
    repairability_pairs,
    repairability_table,
    to_frame,
)
from frontier.analysis.significance import VariantPair
from frontier.io.predictions import (
    CorruptSidecarError,
    LabelArray,
    PredictionRows,
    ProbMatrix,
    pad_option_probs,
    predictions_key,
    predictions_path,
    write_predictions_rows,
)
from frontier.io.store import ResultStore, append_row
from frontier.metrics.bootstrap import ResidualInterval
from frontier.schema import ResultRow

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "io"))

from rows import sample_row

N_ITEMS = 2000
N_VARIANTS = 2
MIN_REMOVED = 0.5
SHARPEN = 0.45  # logits divided by this, so a temperature of 1/SHARPEN undoes it


def _softmax(scores: ProbMatrix) -> ProbMatrix:
    shifted = scores - scores.max(axis=1, keepdims=True)
    exponentiated = np.exp(shifted)
    normalised: ProbMatrix = exponentiated / exponentiated.sum(axis=1, keepdims=True)
    return normalised


def _sidecar(probs: ProbMatrix, gold: LabelArray, qids: list[str]) -> PredictionRows:
    predicted = probs.argmax(axis=1).astype(np.intp)
    return PredictionRows(
        confidence=probs.max(axis=1),
        correct=(predicted == gold),
        gold=gold,
        predicted=predicted,
        options=pad_option_probs(list(probs)),
        qid=np.asarray(qids, dtype=np.str_),
    )


def _store(
    tmp_path: Path, variants: dict[str, float], *, item_ids: dict[str, list[str]] | None = None
) -> tuple[ResultStore, Path]:
    rng = np.random.default_rng(5)
    logits = rng.normal(0.0, 2.0, size=(N_ITEMS, 4))
    gold = np.asarray([rng.choice(4, p=row) for row in _softmax(logits)], dtype=np.intp)
    store = ResultStore(tmp_path)
    for index, (name, sharpen) in enumerate(variants.items()):
        config_hash = str(index) * 64
        base = sample_row()
        row: ResultRow = replace(
            base,
            variant_name=name,
            provenance=replace(base.provenance, config_hash=config_hash),
        )
        append_row(row, store)
        qids = (item_ids or {}).get(name, [f"q{position}" for position in range(N_ITEMS)])
        write_predictions_rows(
            _sidecar(_softmax(logits / sharpen), gold, qids),
            root=tmp_path,
            key=predictions_key(config_hash, 0, "mmlu"),
        )
    return store, tmp_path


def test_a_sharpened_variant_is_repaired_and_a_calibrated_one_is_not(tmp_path: Path) -> None:
    store, root = _store(tmp_path, {"fp16": 1.0, "int4-nf4": SHARPEN})
    found, skipped = repairability_table(load_tidy(store), root=root)
    assert skipped == []
    by_name = {item.variant: item for item in found}
    sharpened = by_name["int4-nf4"]
    assert sharpened.temperature == pytest.approx(1.0 / SHARPEN, rel=0.2)
    assert sharpened.ece_after < sharpened.ece_before
    assert sharpened.ece_removed_fraction > MIN_REMOVED
    assert by_name["fp16"].ece_before < sharpened.ece_before


def test_two_monotone_transforms_of_one_model_leave_the_same_residual(tmp_path: Path) -> None:
    """Both fits search the same one-parameter family, so they must land together.

    The strongest single check on the fit-and-report chain: an argument-order slip or a
    split misalignment breaks it.
    """
    store, root = _store(tmp_path, {"fp16": 1.0, "int4-nf4": SHARPEN})
    found, _ = repairability_table(load_tidy(store), root=root)
    residuals = {item.variant: item.ece_after for item in found}
    assert residuals["fp16"] == pytest.approx(residuals["int4-nf4"], abs=1e-6)


def test_accuracy_is_unchanged_by_the_temperature(tmp_path: Path) -> None:
    """Scaling is monotone within a row, so the argmax cannot move."""
    store, root = _store(tmp_path, {"fp16": 1.0, "int4-nf4": SHARPEN})
    found, _ = repairability_table(load_tidy(store), root=root)
    assert len({round(item.accuracy, 12) for item in found}) == 1


def test_the_fit_half_never_reaches_the_reported_numbers(tmp_path: Path) -> None:
    store, root = _store(tmp_path, {"int4-nf4": SHARPEN})
    rows_fit, rows_report = load_split_predictions(
        load_tidy(store), variant_name="int4-nf4", task_name="mmlu", root=root
    )
    assert rows_fit.qid is not None
    assert rows_report.qid is not None
    assert set(rows_fit.qid).isdisjoint(set(rows_report.qid))
    found, _ = repairability_table(load_tidy(store), root=root)
    assert found[0].n_fit == rows_fit.gold.shape[0]
    assert found[0].n_report == rows_report.gold.shape[0]


def test_a_variant_with_no_stored_distributions_is_skipped(tmp_path: Path) -> None:
    store, root = _store(tmp_path, {"fp16": 1.0})
    key = predictions_key("0" * 64, 0, "mmlu")
    stripped = replace(
        _sidecar(
            _softmax(np.zeros((N_ITEMS, 4))),
            np.zeros(N_ITEMS, dtype=np.intp),
            [f"q{index}" for index in range(N_ITEMS)],
        ),
        options=None,
    )
    write_predictions_rows(stripped, root=root, key=key)
    found, skipped = repairability_table(load_tidy(store), root=root)
    assert found == []
    assert "fp16" in skipped[0].variant


def test_a_corrupt_sidecar_raises_instead_of_being_skipped(tmp_path: Path) -> None:
    """Corruption must not read as an expected absence on a scrolling pod log."""
    store, root = _store(tmp_path, {"fp16": 1.0})
    path = predictions_path(root, predictions_key("0" * 64, 0, "mmlu"))
    table = pq.read_table(path)
    qids = table.column("qid").to_pylist()
    qids[0] = None
    index = table.column_names.index("qid")
    pq.write_table(table.set_column(index, "qid", pa.array(qids, pa.string())), path)
    with pytest.raises(CorruptSidecarError):
        repairability_table(load_tidy(store), root=root)


def test_variants_scored_on_different_items_get_different_fingerprints(tmp_path: Path) -> None:
    store, root = _store(
        tmp_path,
        {"fp16": 1.0, "int4-nf4": SHARPEN},
        item_ids={
            "fp16": [f"q{index}" for index in range(N_ITEMS)],
            "int4-nf4": [f"other{index}" for index in range(N_ITEMS)],
        },
    )
    found, _ = repairability_table(load_tidy(store), root=root)
    assert len({item.report_fingerprint for item in found}) == N_VARIANTS


def test_matching_item_sets_share_a_fingerprint(tmp_path: Path) -> None:
    store, root = _store(tmp_path, {"fp16": 1.0, "int4-nf4": SHARPEN})
    found, _ = repairability_table(load_tidy(store), root=root)
    assert len({item.report_fingerprint for item in found}) == 1


def test_fingerprint_is_order_independent_and_empty_without_ids() -> None:
    ids = np.asarray(["b", "a", "c"], dtype=np.str_)
    assert fingerprint(ids) == fingerprint(np.asarray(["c", "b", "a"], dtype=np.str_))
    assert fingerprint(None) == ""


def test_the_report_half_is_flagged_when_it_falls_under_the_floor(tmp_path: Path) -> None:
    store, root = _store(tmp_path, {"fp16": 1.0})
    found, _ = repairability_table(load_tidy(store), root=root)
    assert found[0].n_report >= REPORT_FLOOR
    assert not found[0].report_below_floor


def test_to_frame_carries_the_bin_count_and_the_fingerprint(tmp_path: Path) -> None:
    store, root = _store(tmp_path, {"fp16": 1.0, "int4-nf4": SHARPEN})
    found, _ = repairability_table(load_tidy(store), root=root, n_bins=15)
    frame = to_frame(found)
    assert len(frame) == N_VARIANTS
    assert set(frame["n_bins"]) == {15}
    assert {
        "variant",
        "n_bins",
        "report_fingerprint",
        "report_below_floor",
        "ece_before",
        "ece_after",
        "ece_removed_fraction",
    } <= set(frame.columns)


def test_an_empty_frame_produces_nothing() -> None:
    found, skipped = repairability_table(pd.DataFrame(), root=Path())
    assert found == []
    assert skipped == []


def test_the_removed_fraction_is_nan_on_a_zero_denominator() -> None:
    degenerate = Repairability(
        variant="v",
        task="mmlu",
        backend="hf",
        family="ptq",
        temperature=1.0,
        n_bins=10,
        n_fit=10,
        n_report=10,
        report_fingerprint="",
        ece_before=0.0,
        ece_after=0.05,
        brier_reliability_before=0.0,
        brier_reliability_after=0.0,
        nll_before=0.0,
        nll_after=0.0,
        accuracy=1.0,
    )
    assert np.isnan(degenerate.ece_removed_fraction)


def test_a_pair_against_the_backend_reference_gets_an_interval(tmp_path: Path) -> None:
    store, root = _store(tmp_path, {"fp16": 1.0, "int4-nf4": SHARPEN})
    found, skipped = repairability_pairs(
        load_tidy(store), root=root, references={"hf": "fp16"}, n_resamples=99
    )
    assert [item.pair.variant for item in found] == ["int4-nf4"]
    assert any("is the reference" in skip.reason for skip in skipped)
    gap = found[0].residual_gap
    assert gap.low <= gap.point <= gap.high
    assert gap.usable


def test_a_monotone_transform_pair_shows_no_difference_beyond_noise(tmp_path: Path) -> None:
    """Both scale into the same distribution, so the gap must not be called a finding."""
    store, root = _store(tmp_path, {"fp16": 1.0, "int4-nf4": SHARPEN})
    found, _ = repairability_pairs(
        load_tidy(store), root=root, references={"hf": "fp16"}, n_resamples=99
    )
    assert not found[0].variant_is_less_repairable
    assert not found[0].residual_gap.excludes_zero


def test_report_halves_over_different_items_are_refused(tmp_path: Path) -> None:
    store, root = _store(
        tmp_path,
        {"fp16": 1.0, "int4-nf4": SHARPEN},
        item_ids={
            "fp16": [f"q{index}" for index in range(N_ITEMS)],
            "int4-nf4": [f"other{index}" for index in range(N_ITEMS)],
        },
    )
    found, skipped = repairability_pairs(
        load_tidy(store), root=root, references={"hf": "fp16"}, n_resamples=19
    )
    assert found == []
    assert any("different items" in skip.reason for skip in skipped)


def test_check_alignment_falls_back_to_length_without_ids() -> None:
    pair = VariantPair(variant="int4-nf4", reference="fp16", task="mmlu", backend="hf", track="A")
    short = PredictionRows(
        np.zeros(3),
        np.zeros(3, dtype=np.bool_),
        np.zeros(3, dtype=np.intp),
        np.zeros(3, dtype=np.intp),
        None,
        None,
    )
    long = PredictionRows(
        np.zeros(4),
        np.zeros(4, dtype=np.bool_),
        np.zeros(4, dtype=np.intp),
        np.zeros(4, dtype=np.intp),
        None,
        None,
    )
    check_alignment(pair, short, short, half="report")
    with pytest.raises(NotRecalibratableError, match="report halves differ in length"):
        check_alignment(pair, short, long, half="report")
    with pytest.raises(NotRecalibratableError, match="fit halves differ in length"):
        check_alignment(pair, short, long, half="fit")


def test_pairs_to_frame_carries_the_interval_and_the_verdict(tmp_path: Path) -> None:
    store, root = _store(tmp_path, {"fp16": 1.0, "int4-nf4": SHARPEN})
    found, _ = repairability_pairs(
        load_tidy(store), root=root, references={"hf": "fp16"}, n_resamples=99
    )
    frame = pairs_to_frame(found)
    assert {
        "variant",
        "reference",
        "residual_variant",
        "residual_reference",
        "residual_gap",
        "residual_gap_low",
        "residual_gap_high",
        "residual_gap_excludes_zero",
        "residual_gap_usable",
        "variant_is_less_repairable",
        "refused_resamples",
    } <= set(frame.columns)


def test_a_sidecar_with_ids_but_no_distributions_is_skipped_in_the_pair_layer(
    tmp_path: Path,
) -> None:
    store, root = _store(tmp_path, {"fp16": 1.0, "int4-nf4": SHARPEN})
    rng = np.random.default_rng(1)
    probs = _softmax(rng.normal(0.0, 2.0, size=(N_ITEMS, 4)))
    predicted = probs.argmax(axis=1).astype(np.intp)
    write_predictions_rows(
        PredictionRows(
            confidence=probs.max(axis=1),
            correct=np.ones(N_ITEMS, dtype=np.bool_),
            gold=predicted,
            predicted=predicted,
            options=None,
            qid=np.asarray([f"q{index}" for index in range(N_ITEMS)], dtype=np.str_),
        ),
        root=root,
        key=predictions_key("1" * 64, 0, "mmlu"),
    )
    found, skipped = repairability_pairs(
        load_tidy(store), root=root, references={"hf": "fp16"}, n_resamples=19
    )
    assert found == []
    assert any("no stored distributions" in skip.reason for skip in skipped)


def _refit_ids(qids: list[str]) -> list[str]:
    """The same report-half ids, with every fit-half id swapped for a different one."""
    spare = (name for name in (f"spare{index}" for index in range(10 * len(qids))) if is_fit(name))
    return [next(spare) if is_fit(qid) else qid for qid in qids]


def test_fit_halves_over_different_items_are_refused(tmp_path: Path) -> None:
    """The report halves match and the fit halves are the same length here.

    Nothing downstream would notice, and the two temperatures come off different questions.
    """
    base = [f"q{index}" for index in range(N_ITEMS)]
    store, root = _store(
        tmp_path,
        {"fp16": 1.0, "int4-nf4": SHARPEN},
        item_ids={"fp16": base, "int4-nf4": _refit_ids(base)},
    )
    found, skipped = repairability_pairs(
        load_tidy(store), root=root, references={"hf": "fp16"}, n_resamples=19
    )
    assert found == []
    assert any("fit halves hold different items" in skip.reason for skip in skipped)


def test_the_pair_residuals_and_their_gap_are_one_number(tmp_path: Path) -> None:
    """The parquet cannot hold a difference that disagrees with the two it sits beside."""
    store, root = _store(tmp_path, {"fp16": 1.0, "int4-nf4": SHARPEN})
    found, _ = repairability_pairs(
        load_tidy(store), root=root, references={"hf": "fp16"}, n_resamples=19
    )
    pair = found[0]
    assert pair.residual_variant - pair.residual_reference == pair.residual_gap.point
    frame = pairs_to_frame(found)
    assert (
        frame["residual_variant"].iloc[0] - frame["residual_reference"].iloc[0]
        == frame["residual_gap"].iloc[0]
    )


def test_a_gap_whose_resamples_partly_refused_is_not_a_verdict() -> None:
    """Those bounds are quantiles over the survivors, so they are finite however few."""
    thinned = RepairabilityPair(
        pair=VariantPair(
            variant="int4-nf4", reference="fp16", task="mmlu", backend="hf", track="A"
        ),
        n_bins=10,
        n_report=1400,
        residual_variant=0.08,
        residual_reference=0.03,
        residual_gap=ResidualInterval(
            point=0.05, low=0.01, high=0.09, refused_resamples=4000, n_resamples=9999
        ),
    )
    assert thinned.residual_gap.excludes_zero
    assert not thinned.variant_is_less_repairable
    assert not pairs_to_frame([thinned])["residual_gap_usable"].iloc[0]


def test_a_variant_spanning_two_config_hashes_is_skipped_not_raised(tmp_path: Path) -> None:
    """A config edit re-run leaves two hashes under one variant. Both layers skip it."""
    store, root = _store(tmp_path, {"fp16": 1.0, "int4-nf4": SHARPEN})
    base = sample_row()
    append_row(
        replace(
            base,
            variant_name="int4-nf4",
            provenance=replace(base.provenance, config_hash="f" * 64),
        ),
        store,
    )
    found, skipped = repairability_table(load_tidy(store), root=root)
    assert [item.variant for item in found] == ["fp16"]
    assert any("config hashes" in skip.reason for skip in skipped)
    paired, pair_skipped = repairability_pairs(
        load_tidy(store), root=root, references={"hf": "fp16"}, n_resamples=19
    )
    assert paired == []
    assert any("config hashes" in skip.reason for skip in pair_skipped)
