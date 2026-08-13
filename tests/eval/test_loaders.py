"""Row normalisers, the MMLU-Redux policy, and the thin loaders on inline fixtures."""

from __future__ import annotations

from collections.abc import Iterable, Mapping

import pytest

from frontier.eval.loaders import (
    load_arc_challenge,
    load_mmlu,
    load_mmlu_redux,
    normalize_arc_row,
    normalize_mmlu_row,
    normalize_redux_row,
    parse_correct_answer,
    redux_gold_arrays,
    resolve_redux_gold,
    subsample,
)
from frontier.eval.records import EvalRecord


def test_normalize_mmlu_row() -> None:
    row = {"question": "2+2?", "subject": "math", "choices": ["3", "4", "5", "6"], "answer": 1}
    record = normalize_mmlu_row(row, index=7, split="test")
    assert record.options == ("3", "4", "5", "6")
    assert record.gold == 1
    assert record.subject == "math"
    assert record.qid == "math:test:7"


def test_normalize_arc_row_letter_labels() -> None:
    row = {
        "id": "arc1",
        "question": "Q?",
        "choices": {"text": ["a", "b", "c", "d"], "label": ["A", "B", "C", "D"]},
        "answerKey": "C",
    }
    record = normalize_arc_row(row, split="test")
    assert record.options[record.gold] == "c"  # answerKey "C" resolves positionally
    assert record.qid == "arc1"
    assert record.subject == "arc_challenge"
    assert record.options == ("a", "b", "c", "d")


def test_normalize_arc_row_numeric_labels() -> None:
    row = {
        "id": "arc2",
        "question": "Q?",
        "choices": {"text": ["a", "b", "c", "d"], "label": ["1", "2", "3", "4"]},
        "answerKey": "3",
    }
    record = normalize_arc_row(row, split="test")
    assert record.options[record.gold] == "c"  # numeric label "3" resolves to the third option
    assert record.options == ("a", "b", "c", "d")


def test_normalize_arc_row_three_and_five_options() -> None:
    three = {
        "id": "arc3",
        "question": "Q?",
        "choices": {"text": ["a", "b", "c"], "label": ["A", "B", "C"]},
        "answerKey": "B",
    }
    five = {
        "id": "arc5",
        "question": "Q?",
        "choices": {"text": ["a", "b", "c", "d", "e"], "label": ["A", "B", "C", "D", "E"]},
        "answerKey": "E",
    }
    three_record = normalize_arc_row(three, split="test")
    five_record = normalize_arc_row(five, split="test")
    assert three_record.options == ("a", "b", "c")
    assert three_record.options[three_record.gold] == "b"
    assert five_record.options == ("a", "b", "c", "d", "e")
    assert five_record.options[five_record.gold] == "e"


def test_normalize_arc_row_rejects_bad_answer_key() -> None:
    row = {
        "id": "arcx",
        "question": "Q?",
        "choices": {"text": ["a", "b"], "label": ["A", "B"]},
        "answerKey": "Z",
    }
    with pytest.raises(ValueError, match=r"'arcx'.*answerKey"):
        normalize_arc_row(row, split="test")


def test_normalize_arc_row_rejects_mismatched_lengths() -> None:
    row = {
        "id": "arcy",
        "question": "Q?",
        "choices": {"text": ["a", "b", "c"], "label": ["A", "B"]},
        "answerKey": "A",
    }
    with pytest.raises(ValueError, match=r"'arcy'.*3 options but 2 labels"):
        normalize_arc_row(row, split="test")


def test_parse_correct_answer() -> None:
    options = ["apple", "Banana", "cherry", "date"]
    assert parse_correct_answer("C", options) == options.index("cherry")
    assert parse_correct_answer("3", options) == options.index("cherry")
    assert parse_correct_answer("  banana ", options) == options.index("Banana")
    assert parse_correct_answer("zzz", options) is None
    assert parse_correct_answer("9", options) is None


def test_parse_correct_answer_precedence_is_pinned() -> None:
    # A lone digit is read positionally (digit - 1), so "3" here means the third option.
    digit_options = ["3", "4", "5", "6"]
    assert parse_correct_answer("3", digit_options) == digit_options.index("5")
    # A lone letter is read positionally too, so an out-of-range letter drops the item.
    assert parse_correct_answer("X", ["X", "Y"]) is None


def test_resolve_redux_gold_policies() -> None:
    options = ["a", "b", "c", "d"]
    raw_gold = 1
    assert resolve_redux_gold("ok", "", options, raw_gold, policy="clean_subset") == raw_gold
    assert resolve_redux_gold("ok", "", options, raw_gold, policy="relabel") == raw_gold
    assert (
        resolve_redux_gold("wrong_groundtruth", "C", options, raw_gold, policy="clean_subset")
        is None
    )
    assert resolve_redux_gold(
        "wrong_groundtruth", "C", options, raw_gold, policy="relabel"
    ) == options.index("c")
    assert (
        resolve_redux_gold("bad_question_clarity", "", options, raw_gold, policy="relabel") is None
    )
    assert resolve_redux_gold("expert", "", options, raw_gold, policy="clean_subset") is None


def _redux_rows() -> list[dict[str, object]]:
    return [
        {
            "question": "q0",
            "choices": ["a", "b", "c", "d"],
            "answer": 1,
            "error_type": "ok",
            "correct_answer": "",
        },
        {
            "question": "q1",
            "choices": ["a", "b", "c", "d"],
            "answer": 0,
            "error_type": "wrong_groundtruth",
            "correct_answer": "C",
        },
        {
            "question": "q2",
            "choices": ["a", "b", "c", "d"],
            "answer": 2,
            "error_type": "bad_options_clarity",
            "correct_answer": "",
        },
    ]


def test_normalize_redux_row_carries_both_golds_under_relabel() -> None:
    records = [
        normalize_redux_row(row, subject="math", index=i, policy="relabel")
        for i, row in enumerate(_redux_rows())
    ]
    assert [record.gold for record in records] == [1, 0, 2]
    assert [record.redux_gold for record in records] == [1, 2, None]
    assert records[0].error_type == "ok"
    raw_gold, redux_on_kept, keep_mask = redux_gold_arrays(records)
    assert raw_gold.tolist() == [1, 0, 2]
    assert keep_mask.tolist() == [True, True, False]
    assert redux_on_kept.tolist() == [1, 2]


def test_normalize_redux_row_clean_subset_keeps_only_ok() -> None:
    records = [
        normalize_redux_row(row, subject="math", index=i, policy="clean_subset")
        for i, row in enumerate(_redux_rows())
    ]
    assert [record.redux_gold for record in records] == [1, None, None]


def _fake_mmlu_loader(
    path: str, name: str | None = None, *, split: str
) -> Iterable[Mapping[str, object]]:
    assert path == "cais/mmlu"
    assert name == "all"
    assert split == "test"
    return [
        {"question": "q0", "subject": "math", "choices": ["3", "4", "5", "6"], "answer": 1},
        {"question": "q1", "subject": "history", "choices": ["w", "x", "y", "z"], "answer": 3},
    ]


def test_thin_mmlu_loader_assembles_records_without_network() -> None:
    rows = list(_fake_mmlu_loader("cais/mmlu", "all", split="test"))
    records = load_mmlu(split="test", loader=_fake_mmlu_loader)
    assert len(records) == len(rows)
    assert records[0].qid == "math:test:0"
    assert records[1].gold == rows[1]["answer"]


def _fake_arc_loader(
    path: str, name: str | None = None, *, split: str
) -> Iterable[Mapping[str, object]]:
    assert path == "allenai/ai2_arc"
    assert name == "ARC-Challenge"
    assert split == "test"
    return [
        {
            "id": "arcA",
            "question": "Q0?",
            "choices": {"text": ["a", "b", "c", "d"], "label": ["A", "B", "C", "D"]},
            "answerKey": "B",
        },
        {
            "id": "arcB",
            "question": "Q1?",
            "choices": {"text": ["a", "b", "c"], "label": ["1", "2", "3"]},
            "answerKey": "3",
        },
    ]


def test_thin_arc_loader_assembles_records_without_network() -> None:
    records = load_arc_challenge(split="test", loader=_fake_arc_loader)
    assert [record.qid for record in records] == ["arcA", "arcB"]
    assert records[0].options[records[0].gold] == "b"
    assert records[1].options[records[1].gold] == "c"


def _fake_redux_loader(
    path: str, name: str | None = None, *, split: str
) -> Iterable[Mapping[str, object]]:
    assert path == "edinburgh-dawg/mmlu-redux-2.0"
    assert isinstance(name, str)  # the subject config name
    assert split == "test"
    return _redux_rows()


def test_thin_redux_loader_iterates_subject_configs() -> None:
    configs = ["anatomy", "astronomy"]
    records = load_mmlu_redux(
        policy="relabel", loader=_fake_redux_loader, config_names=lambda _path: configs
    )
    assert [record.subject for record in records[:3]] == ["anatomy", "anatomy", "anatomy"]
    assert records[0].qid == "redux:anatomy:0"
    assert records[len(_redux_rows())].subject == "astronomy"
    assert len(records) == len(configs) * len(_redux_rows())


def test_normalize_mmlu_row_rejects_non_integer_answer() -> None:
    row = {"question": "q", "subject": "s", "choices": ["a", "b", "c", "d"], "answer": "1"}
    with pytest.raises(TypeError, match="integer answer index"):
        normalize_mmlu_row(row, index=0, split="test")


def test_normalize_mmlu_row_rejects_boolean_answer() -> None:
    row = {"question": "q", "subject": "s", "choices": ["a", "b", "c", "d"], "answer": True}
    with pytest.raises(TypeError, match="bool"):
        normalize_mmlu_row(row, index=0, split="test")


def test_normalize_mmlu_row_rejects_non_sequence_choices() -> None:
    row = {"question": "q", "subject": "s", "choices": 4, "answer": 1}
    with pytest.raises(TypeError, match="sequence of option strings"):
        normalize_mmlu_row(row, index=0, split="test")


def test_normalize_arc_row_rejects_non_mapping_choices() -> None:
    row = {"id": "arcz", "question": "q", "choices": ["a", "b"], "answerKey": "A"}
    with pytest.raises(TypeError, match=r"'arcz'.*non-mapping"):
        normalize_arc_row(row, split="test")


def _subject_records() -> list[EvalRecord]:
    subjects = ["a"] * 6 + ["b"] * 4
    return [
        EvalRecord(
            qid=f"q{i}",
            question="Q?",
            options=("a", "b", "c", "d"),
            gold=0,
            subject=subject,
            split="test",
        )
        for i, subject in enumerate(subjects)
    ]


def test_subsample_is_deterministic_and_stratified() -> None:
    records = _subject_records()
    size = 4
    first = subsample(records, size, seed=0)
    second = subsample(records, size, seed=0)
    assert [record.qid for record in first] == [record.qid for record in second]
    assert len(first) == size
    assert {record.subject for record in first} == {"a", "b"}


def test_subsample_returns_all_when_size_exceeds_pool() -> None:
    records = _subject_records()
    assert len(subsample(records, len(records) + 5, seed=0)) == len(records)
