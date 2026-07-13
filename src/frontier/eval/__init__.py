"""Task evaluation: dataset loaders (MMLU / MMLU-Redux / ARC), prompt building,
answer-letter confidence extraction with cyclic permutation debiasing, and
exact-match correctness. Produces the per-item probability rows and gold labels the
metrics package consumes.

The package is model-agnostic and CPU-only. It obtains next-token logits through the
``LogitProvider`` protocol; the real Hugging Face backend that produces them is a
later work package.
"""

from __future__ import annotations

from frontier.eval.extract import (
    EvalOutputs,
    PermutationRobustness,
    exact_match,
    score_items,
    to_robustness,
    to_task_spec,
)
from frontier.eval.loaders import (
    REDUX_DROP_ALWAYS,
    TASK_LOADERS,
    ReduxPolicy,
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
from frontier.eval.prompts import (
    ANSWER_TRIGGER,
    COT_TRIGGER,
    INSTRUCTION,
    build_prompt,
    options_block,
)
from frontier.eval.provider import (
    LogitProvider,
    Tokenizer,
    letter_probs,
    resolve_candidate_ids,
    softmax,
)
from frontier.eval.records import LETTERS, MAX_OPTIONS, EvalRecord, letters_for

__all__ = [
    "ANSWER_TRIGGER",
    "COT_TRIGGER",
    "INSTRUCTION",
    "LETTERS",
    "MAX_OPTIONS",
    "REDUX_DROP_ALWAYS",
    "TASK_LOADERS",
    "EvalOutputs",
    "EvalRecord",
    "LogitProvider",
    "PermutationRobustness",
    "ReduxPolicy",
    "Tokenizer",
    "build_prompt",
    "exact_match",
    "letter_probs",
    "letters_for",
    "load_arc_challenge",
    "load_mmlu",
    "load_mmlu_redux",
    "normalize_arc_row",
    "normalize_mmlu_row",
    "normalize_redux_row",
    "options_block",
    "parse_correct_answer",
    "redux_gold_arrays",
    "resolve_candidate_ids",
    "resolve_redux_gold",
    "score_items",
    "softmax",
    "subsample",
    "to_robustness",
    "to_task_spec",
]
