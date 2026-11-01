"""The fidelity check passes a clean cache and catches each way a build can go wrong."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from tiny import arithmetic_rows, build_cache, successor_forward

from frontier.distill.cache import REST_FILE, TOPK_IDS_FILE, TOPK_PROBS_FILE
from frontier.distill.fidelity import (
    FIDELITY_FILE,
    CacheFidelityError,
    check_fidelity,
    read_fidelity,
)

torch = pytest.importorskip("torch")

N_ROWS = 10  # five two-row batches; four are checked, the middle one is skipped
TOP_K = 3
CHECKED_ROWS = (0, 1, 2, 3, 6, 7, 8, 9)
PROB_SCALE = 1.01
REST_BUMP = 1e-3


def _clean(out: Path, forward: Any = None) -> Any:
    teacher = successor_forward()
    build_cache(out, arithmetic_rows(N_ROWS), forward or teacher, top_k=TOP_K)
    return teacher


def _edit(out: Path, name: str, change: Callable[[Any], None]) -> None:
    stored = np.load(out / name, mmap_mode="r+")
    change(stored)
    stored.flush()


def test_a_clean_cache_passes_and_peaks_at_offset_one(tmp_path: Path) -> None:
    teacher = _clean(tmp_path)
    report = check_fidelity(tmp_path, teacher, device="cpu")
    assert report.passed
    assert report.first_failure is None
    assert report.rows == CHECKED_ROWS
    assert report.positions == len(CHECKED_ROWS) * (arithmetic_rows(1).shape[1] - 1)
    assert report.next_token_agreement == {0: 0.0, 1: 1.0, 2: 0.0}
    assert read_fidelity(tmp_path) == report


def test_one_batch_checks_the_first_alone(tmp_path: Path) -> None:
    assert check_fidelity(tmp_path, _clean(tmp_path), device="cpu", batches=1).rows == (0, 1)


def _swap_rows(stored: Any) -> None:
    stored[[0, 1]] = stored[[1, 0]]


def _scale_one_probability(stored: Any) -> None:
    stored[8, 3, 0] *= PROB_SCALE


def _raise_one_rest(stored: Any) -> None:
    stored[9, 2] += REST_BUMP


def _swap_first_two_columns(stored: Any) -> None:
    stored[8, 2, [0, 1]] = stored[8, 2, [1, 0]]


@pytest.mark.parametrize(
    ("edits", "first_failure"),
    [
        (
            [(name, _swap_rows) for name in (TOPK_IDS_FILE, TOPK_PROBS_FILE, REST_FILE)],
            "row 0 position 0",
        ),
        ([(TOPK_PROBS_FILE, _scale_one_probability)], "row 8 position 3: cached probability"),
        ([(REST_FILE, _raise_one_rest)], "row 9 position 2: cached rest"),
        (
            [(TOPK_IDS_FILE, _swap_first_two_columns), (TOPK_PROBS_FILE, _swap_first_two_columns)],
            "row 8 position 2: cached probabilities increase",
        ),
    ],
    ids=["rows-swapped", "probability-scaled", "rest-raised", "order-swapped"],
)
def test_a_corrupted_cache_fails_and_names_the_first_bad_position(
    tmp_path: Path, edits: list[tuple[str, Callable[[Any], None]]], first_failure: str
) -> None:
    teacher = _clean(tmp_path)
    for name, change in edits:
        _edit(tmp_path, name, change)
    with pytest.raises(CacheFidelityError, match=first_failure):
        check_fidelity(tmp_path, teacher, device="cpu")
    stored = json.loads((tmp_path / FIDELITY_FILE).read_text(encoding="utf-8"))
    assert stored["passed"] is False
    assert stored["first_failure"].startswith(first_failure)


def test_an_id_below_the_k_th_largest_is_a_top_k_violation(tmp_path: Path) -> None:
    teacher = _clean(tmp_path)
    row, position = 6, 4
    z = teacher(torch.from_numpy(arithmetic_rows(N_ROWS)))[row, position].double()
    p64 = torch.softmax(z, dim=-1)
    lowest = int(p64.argmin())

    def replace_last_id(stored: Any) -> None:
        stored[row, position, TOP_K - 1] = lowest

    def replace_last_probability(stored: Any) -> None:
        stored[row, position, TOP_K - 1] = float(p64[lowest])

    _edit(tmp_path, TOPK_IDS_FILE, replace_last_id)
    _edit(tmp_path, TOPK_PROBS_FILE, replace_last_probability)
    with pytest.raises(CacheFidelityError, match=f"row {row} position {position}: a cached id"):
        check_fidelity(tmp_path, teacher, device="cpu")
    assert read_fidelity(tmp_path).topk_violations == 1


def test_a_cache_written_one_position_late_fails(tmp_path: Path) -> None:
    teacher = successor_forward()

    def shifted(ids: Any) -> Any:
        logits = teacher(ids)
        return torch.cat([logits[:, 1:], logits[:, -1:]], dim=1)

    _clean(tmp_path, shifted)
    with pytest.raises(CacheFidelityError, match="row 0 position 0"):
        check_fidelity(tmp_path, teacher, device="cpu")
    agreement = read_fidelity(tmp_path).next_token_agreement
    assert agreement[2] > agreement[1]
