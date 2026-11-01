"""The cache fidelity check: recompute whole build batches and compare against an fp64
softmax computed apart from ``teacher_targets``, so a build bug cannot hide in shared code."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from frontier.distill.cache import (
    INPUT_IDS_FILE,
    REST_FILE,
    TOPK_IDS_FILE,
    TOPK_PROBS_FILE,
    CacheMeta,
    Forward,
    read_meta,
    write_json,
)

FIDELITY_FILE = "fidelity.json"
FIDELITY_BATCHES = 4
PROB_RTOL = 2**-11 + 1e-5  # fp16 half-ulp, relative, plus fp32 softmax rounding
PROB_ATOL = 2**-24  # one fp16 subnormal step
REST_RTOL = 1e-4  # fp32 logsumexp over a 151,936-entry vocabulary, relative
REST_ATOL = 1e-12
NEXT_TOKEN_OFFSETS = (0, 1, 2)


@dataclass(frozen=True, slots=True)
class FidelityReport:
    """The recompute against the cache, over ``rows``.

    A ratio is the worst error over its tolerance, so it passes at or below 1.
    ``next_token_agreement`` maps an offset to the fraction of positions whose cached top-1
    is the input token that far ahead; a correctly aligned cache peaks at offset 1.
    """

    rows: tuple[int, ...]
    positions: int
    prob_error_ratio: float  # max |p_cached - p64| / (PROB_RTOL * p64 + PROB_ATOL)
    prob_abs_error: float
    rest_error_ratio: float  # max |rest_cached - rest64| / (REST_RTOL * rest64 + REST_ATOL)
    rest_abs_error: float
    topk_violations: int  # positions where a cached id is below the k-th largest p64
    order_violations: int  # positions whose cached probabilities increase along k
    duplicate_ids: int  # positions with a repeated id
    next_token_agreement: dict[int, float]
    rest_mean: float
    rest_max: float
    first_failure: str | None  # "row r position t: ..." for the first failing check

    @property
    def passed(self) -> bool:
        return (
            self.prob_error_ratio <= 1.0
            and self.rest_error_ratio <= 1.0
            and self.topk_violations == 0
            and self.order_violations == 0
            and self.duplicate_ids == 0
        )


class CacheFidelityError(RuntimeError):
    """The cache disagrees with a recompute of the teacher."""


def check_fidelity(
    out: Path, forward: Forward, *, device: str, batches: int = FIDELITY_BATCHES
) -> FidelityReport:
    """Recompute ``batches`` build batches, compare, and write ``fidelity.json``.

    The batches are the first, the last, and evenly spaced between, over the rows filled so
    far; ``batches=1`` checks the first alone. Raises ``CacheFidelityError`` unless the report
    passes, after writing it.
    """
    import torch  # noqa: PLC0415

    meta = read_meta(out)
    input_ids = np.load(out / INPUT_IDS_FILE, mmap_mode="r")
    topk_ids = np.load(out / TOPK_IDS_FILE, mmap_mode="r")
    topk_probs = np.load(out / TOPK_PROBS_FILE, mmap_mode="r")
    rest = np.load(out / REST_FILE, mmap_mode="r")
    tally = _Tally()
    checked: list[int] = []
    for first in _batch_starts(meta, batches):
        last = min(first + meta.batch_rows, meta.rows_done)
        ids = torch.from_numpy(input_ids[first:last].astype(np.int64)).to(device)
        with torch.inference_mode():
            logits = forward(ids)
            for offset, row in enumerate(range(first, last)):
                tally.add_row(
                    row,
                    logits[offset, : meta.seq_len - 1].double(),
                    cached_ids=torch.from_numpy(topk_ids[row].astype(np.int64)).to(device),
                    cached_probs=torch.from_numpy(topk_probs[row].astype(np.float64)).to(device),
                    cached_rest=torch.from_numpy(rest[row].astype(np.float64)).to(device),
                    tokens=input_ids[row].astype(np.int64),
                )
                checked.append(row)
    report = tally.report(tuple(checked))
    write_json(out / FIDELITY_FILE, {**asdict(report), "passed": report.passed})
    if not report.passed:
        raise CacheFidelityError(
            f"teacher cache {out} disagrees with a recompute of the teacher: "
            f"{report.first_failure}; probability error {report.prob_error_ratio:.3g}x and rest "
            f"error {report.rest_error_ratio:.3g}x tolerance (report in {FIDELITY_FILE})"
        )
    return report


def read_fidelity(out: Path) -> FidelityReport:
    """The last ``fidelity.json`` written for the cache at ``out``."""
    stored = json.loads((out / FIDELITY_FILE).read_text(encoding="utf-8"))
    stored.pop("passed")
    stored["rows"] = tuple(stored["rows"])
    stored["next_token_agreement"] = {
        int(offset): float(share) for offset, share in stored["next_token_agreement"].items()
    }
    return FidelityReport(**stored)


def _batch_starts(meta: CacheMeta, batches: int) -> list[int]:
    filled = math.ceil(meta.rows_done / meta.batch_rows)
    if filled == 0:
        raise ValueError("the cache holds no filled rows to check")
    if batches == 1 or filled == 1:
        return [0]
    picked = {round(i * (filled - 1) / (batches - 1)) for i in range(batches)}
    return [index * meta.batch_rows for index in sorted(picked)]


class _Tally:
    """Worst errors, violation counts, and agreement counts over the checked rows."""

    def __init__(self) -> None:
        self.positions = 0
        self.prob_ratio = self.prob_abs = self.rest_ratio = self.rest_abs = 0.0
        self.topk = self.order = self.duplicates = 0
        self.agree = dict.fromkeys(NEXT_TOKEN_OFFSETS, 0)
        self.agree_total = dict.fromkeys(NEXT_TOKEN_OFFSETS, 0)
        self.rest_sum = self.rest_max = 0.0
        self.first_failure: str | None = None

    def add_row(
        self,
        row: int,
        z: Any,
        *,
        cached_ids: Any,
        cached_probs: Any,
        cached_rest: Any,
        tokens: npt.NDArray[np.int64],
    ) -> None:
        import torch  # noqa: PLC0415

        lse = torch.logsumexp(z, dim=-1)
        p64 = torch.exp(z.gather(-1, cached_ids) - lse[:, None])
        prob_error = (cached_probs - p64).abs()
        prob_ratio = (prob_error / (PROB_RTOL * p64 + PROB_ATOL)).amax(-1)
        kth = torch.exp(torch.topk(z, cached_ids.shape[-1], dim=-1).values[:, -1] - lse)
        below_kth = p64.amin(-1) < kth
        increasing = (cached_probs[:, 1:] > cached_probs[:, :-1]).any(-1)
        ordered_ids = cached_ids.sort(-1).values
        repeated = (ordered_ids[:, 1:] == ordered_ids[:, :-1]).any(-1)
        rest64 = torch.exp(torch.logsumexp(z.scatter(-1, cached_ids, float("-inf")), -1) - lse)
        rest_error = (cached_rest - rest64).abs()
        rest_ratio = rest_error / (REST_RTOL * rest64 + REST_ATOL)

        self.positions += int(z.shape[0])
        self.prob_ratio = max(self.prob_ratio, float(prob_ratio.max()))
        self.prob_abs = max(self.prob_abs, float(prob_error.max()))
        self.rest_ratio = max(self.rest_ratio, float(rest_ratio.max()))
        self.rest_abs = max(self.rest_abs, float(rest_error.max()))
        self.topk += int(below_kth.sum())
        self.order += int(increasing.sum())
        self.duplicates += int(repeated.sum())
        self.rest_sum += float(cached_rest.sum())
        self.rest_max = max(self.rest_max, float(cached_rest.max()))
        top1 = cached_ids[:, 0].cpu().numpy()
        for offset in NEXT_TOKEN_OFFSETS:
            span = min(top1.shape[0], tokens.shape[0] - offset)
            self.agree[offset] += int((top1[:span] == tokens[offset : offset + span]).sum())
            self.agree_total[offset] += span
        if self.first_failure is None:
            for failing, what in (
                (prob_ratio > 1.0, "cached probability outside tolerance"),
                (below_kth, "a cached id sits below the k-th largest probability"),
                (increasing, "cached probabilities increase along k"),
                (repeated, "a cached id repeats"),
                (rest_ratio > 1.0, "cached rest outside tolerance"),
            ):
                if bool(failing.any()):
                    position = int(torch.nonzero(failing)[0, 0])
                    self.first_failure = f"row {row} position {position}: {what}"
                    break

    def report(self, rows: tuple[int, ...]) -> FidelityReport:
        return FidelityReport(
            rows=rows,
            positions=self.positions,
            prob_error_ratio=self.prob_ratio,
            prob_abs_error=self.prob_abs,
            rest_error_ratio=self.rest_ratio,
            rest_abs_error=self.rest_abs,
            topk_violations=self.topk,
            order_violations=self.order,
            duplicate_ids=self.duplicates,
            next_token_agreement={
                offset: self.agree[offset] / self.agree_total[offset]
                for offset in NEXT_TOKEN_OFFSETS
            },
            rest_mean=self.rest_sum / self.positions,
            rest_max=self.rest_max,
            first_failure=self.first_failure,
        )
