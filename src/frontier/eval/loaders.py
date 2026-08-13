"""Row normalisers, the MMLU-Redux de-noise policy, and the thin Hugging Face loaders.

``datasets`` >= 4.0 removed the loading scripts, so these read each repo's published parquet
and pass no ``trust_remote_code``; a script-only repo would need the auto-converted branch,
``revision="refs/convert/parquet"``.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Literal

import numpy as np
import numpy.typing as npt

from frontier.eval.records import EvalRecord, LabelArray

DatasetLoader = Callable[..., Iterable[Mapping[str, object]]]
ConfigNames = Callable[[str], Sequence[str]]

REDUX_DROP_ALWAYS = frozenset(
    {
        "bad_question_clarity",
        "bad_options_clarity",
        "no_correct_answer",
        "multiple_correct_answers",
        "expert",
    }
)
ReduxPolicy = Literal["clean_subset", "relabel"]


def _as_int(value: object) -> int:
    if isinstance(value, bool):
        raise TypeError(f"expected an integer answer index, got a bool: {value!r}")
    if isinstance(value, (int, np.integer)):
        return int(value)
    raise TypeError(f"expected an integer answer index, got {type(value).__name__}: {value!r}")


def _as_str_tuple(value: object) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"expected a sequence of option strings, got {type(value).__name__}")
    return tuple(str(item) for item in value)


def normalize_mmlu_row(row: Mapping[str, object], *, index: int, split: str) -> EvalRecord:
    """A ``cais/mmlu`` ``all`` row to an ``EvalRecord``, with ``qid`` as ``subject:split:index``."""
    subject = str(row["subject"])
    return EvalRecord(
        qid=f"{subject}:{split}:{index}",
        question=str(row["question"]),
        options=_as_str_tuple(row["choices"]),
        gold=_as_int(row["answer"]),
        subject=subject,
        split=split,
    )


def normalize_arc_row(row: Mapping[str, object], *, split: str) -> EvalRecord:
    """An ``ai2_arc`` ARC-Challenge row to an ``EvalRecord``, with gold resolved positionally.

    ``labels.index(answerKey)`` covers the letter and numeric label sets alike, and any 3- or
    5-option item, since the scorer re-letters by position.
    """
    qid = str(row["id"])
    choices = row["choices"]
    if not isinstance(choices, Mapping):
        raise TypeError(f"ARC item {qid!r} has a non-mapping 'choices' field")
    text = _as_str_tuple(choices["text"])
    labels = _as_str_tuple(choices["label"])
    if len(text) != len(labels):
        raise ValueError(f"ARC item {qid!r} has {len(text)} options but {len(labels)} labels")
    answer_key = str(row["answerKey"])
    if answer_key not in labels:
        raise ValueError(f"ARC item {qid!r} answerKey {answer_key!r} not in labels {list(labels)}")
    return EvalRecord(
        qid=qid,
        question=str(row["question"]),
        options=text,
        gold=labels.index(answer_key),
        subject="arc_challenge",
        split=split,
    )


def parse_correct_answer(value: str, options: Sequence[str]) -> int | None:
    """Map an MMLU-Redux ``correct_answer`` string to an option index, or ``None``.

    The field's encoding varies across the corpus, so three forms are tried in order: a lone
    letter, a lone 1-based digit, then a case-insensitive match against an option string. An
    unreadable value or an out-of-range index returns ``None``, and the item is dropped.
    """
    stripped = value.strip()
    if len(stripped) == 1 and stripped.isalpha():
        index = ord(stripped.upper()) - ord("A")
        return index if 0 <= index < len(options) else None
    if len(stripped) == 1 and stripped.isdigit():
        index = int(stripped) - 1
        return index if 0 <= index < len(options) else None
    needle = stripped.casefold()
    for i, option in enumerate(options):
        if option.strip().casefold() == needle:
            return i
    return None


def resolve_redux_gold(
    error_type: str,
    correct_answer: str,
    options: Sequence[str],
    raw_gold: int,
    *,
    policy: ReduxPolicy,
) -> int | None:
    """The de-noised gold for one MMLU-Redux item, or ``None`` to drop it.

    An ``ok`` item keeps its raw gold under either policy. ``clean_subset`` drops every other
    item; ``relabel`` re-labels ``wrong_groundtruth`` from ``correct_answer`` and drops the
    rest, including any unrecognised error type.
    """
    if error_type == "ok":
        return raw_gold
    if policy == "clean_subset":
        return None
    if error_type == "wrong_groundtruth":
        return parse_correct_answer(correct_answer, options)
    return None


def normalize_redux_row(
    row: Mapping[str, object], *, subject: str, index: int, policy: ReduxPolicy
) -> EvalRecord:
    """An ``mmlu-redux-2.0`` row to an ``EvalRecord`` carrying both golds.

    ``gold`` stays the original MMLU label and ``redux_gold`` holds the de-noised one, so one
    record supports the raw-vs-redux comparison.
    """
    options = _as_str_tuple(row["choices"])
    raw_gold = _as_int(row["answer"])
    error_type = str(row["error_type"])
    raw_correct = row.get("correct_answer")
    correct_answer = raw_correct if isinstance(raw_correct, str) else ""
    return EvalRecord(
        qid=f"redux:{subject}:{index}",
        question=str(row["question"]),
        options=options,
        gold=raw_gold,
        subject=subject,
        split="test",
        error_type=error_type,
        redux_gold=resolve_redux_gold(error_type, correct_answer, options, raw_gold, policy=policy),
    )


def redux_gold_arrays(
    records: Sequence[EvalRecord],
) -> tuple[LabelArray, LabelArray, npt.NDArray[np.bool_]]:
    """Return ``(raw_gold, redux_gold_on_kept, keep_mask)`` for the raw-vs-redux comparison.

    ``redux_gold_on_kept`` is already filtered by ``keep_mask``, so it is the shorter array
    and pairing it with ``raw_gold`` needs the mask applied first. The delta is taken on the
    kept items alone, so it isolates the label flip from the item drop.
    """
    raw_gold = np.asarray([record.gold for record in records], dtype=np.intp)
    keep_mask = np.asarray([record.redux_gold is not None for record in records], dtype=np.bool_)
    redux_on_kept = np.asarray(
        [record.redux_gold for record in records if record.redux_gold is not None], dtype=np.intp
    )
    return raw_gold, redux_on_kept, keep_mask


def _load_split(
    path: str, name: str | None = None, *, split: str
) -> Iterable[Mapping[str, object]]:  # pragma: no cover
    import datasets  # noqa: PLC0415

    rows: Iterable[Mapping[str, object]] = datasets.load_dataset(path, name, split=split)
    return rows


def _config_names(path: str) -> Sequence[str]:  # pragma: no cover
    import datasets  # noqa: PLC0415

    names: Sequence[str] = datasets.get_dataset_config_names(path)
    return names


def load_mmlu(
    split: str = "test",
    *,
    subset: int | None = None,
    seed: int = 0,
    loader: DatasetLoader | None = None,
) -> list[EvalRecord]:
    """Load ``cais/mmlu`` config ``all`` and normalise it, subsampling if asked."""
    load = loader or _load_split
    records = [
        normalize_mmlu_row(row, index=i, split=split)
        for i, row in enumerate(load("cais/mmlu", "all", split=split))
    ]
    return records if subset is None else subsample(records, subset, seed=seed)


def load_arc_challenge(
    split: str = "test",
    *,
    subset: int | None = None,
    seed: int = 0,
    loader: DatasetLoader | None = None,
) -> list[EvalRecord]:
    """Load ``allenai/ai2_arc`` config ARC-Challenge and normalise it."""
    load = loader or _load_split
    records = [
        normalize_arc_row(row, split=split)
        for row in load("allenai/ai2_arc", "ARC-Challenge", split=split)
    ]
    return records if subset is None else subsample(records, subset, seed=seed)


def load_mmlu_redux(
    *,
    policy: ReduxPolicy = "clean_subset",
    subjects: Sequence[str] | None = None,
    subset: int | None = None,
    seed: int = 0,
    loader: DatasetLoader | None = None,
    config_names: ConfigNames | None = None,
) -> list[EvalRecord]:
    """Load ``edinburgh-dawg/mmlu-redux-2.0`` across its subject configs and concatenate.

    The repo has no ``all`` config, so the subject configs are iterated and their ``test``
    splits concatenated. ``policy`` is applied per row by ``resolve_redux_gold``.
    """
    load = loader or _load_split
    resolve_names = config_names or _config_names
    names = subjects if subjects is not None else resolve_names("edinburgh-dawg/mmlu-redux-2.0")
    records = [
        normalize_redux_row(row, subject=subject, index=i, policy=policy)
        for subject in names
        for i, row in enumerate(load("edinburgh-dawg/mmlu-redux-2.0", subject, split="test"))
    ]
    return records if subset is None else subsample(records, subset, seed=seed)


TASK_LOADERS: dict[str, Callable[..., list[EvalRecord]]] = {
    "mmlu": load_mmlu,
    "arc_challenge": load_arc_challenge,
    "mmlu_redux": load_mmlu_redux,
}


def subsample(
    records: Sequence[EvalRecord], size: int, *, seed: int, stratify_by: str = "subject"
) -> list[EvalRecord]:
    """Deterministic stratified subsample by subject, with proportional allocation.

    Rounding leftovers go to the largest fractional parts. ``size >= len(records)`` returns
    every record, in the original order.
    """
    n = len(records)
    if size >= n:
        return list(records)
    if size <= 0:
        raise ValueError(f"subsample size must be positive, got {size}")

    rng = np.random.default_rng(seed)
    groups: dict[str, list[int]] = {}
    for i, record in enumerate(records):
        groups.setdefault(str(getattr(record, stratify_by)), []).append(i)

    keys = sorted(groups)
    stratum_sizes = np.asarray([len(groups[key]) for key in keys], dtype=np.float64)
    exact = stratum_sizes / n * size
    allocation = np.floor(exact).astype(np.intp)
    remainder = size - int(allocation.sum())
    for key_index in np.argsort(-(exact - allocation), kind="stable")[:remainder]:
        allocation[key_index] += 1

    selected: list[int] = []
    for key, count in zip(keys, allocation, strict=True):
        indices = groups[key]
        take = min(int(count), len(indices))
        chosen = rng.choice(len(indices), size=take, replace=False)
        selected.extend(indices[j] for j in sorted(int(c) for c in chosen))
    selected.sort()
    return [records[i] for i in selected]
