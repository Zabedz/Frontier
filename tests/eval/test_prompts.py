"""Exact prompt text for the zero-shot and CoT forms."""

from __future__ import annotations

from frontier.eval.prompts import ANSWER_TRIGGER, COT_TRIGGER, build_prompt, options_block


def test_build_prompt_non_cot_is_exact_and_ends_in_answer_trigger() -> None:
    prompt = build_prompt("What is 2+2?", ["3", "4", "5", "6"])
    expected = (
        "Answer the following multiple choice question with the letter of the correct option.\n"
        "\n"
        "Question: What is 2+2?\n"
        "A. 3\n"
        "B. 4\n"
        "C. 5\n"
        "D. 6\n"
        "Answer:"
    )
    assert prompt == expected
    assert prompt.endswith(ANSWER_TRIGGER)
    assert not prompt.endswith("Answer: ")


def test_build_prompt_cot_ends_in_trigger_and_omits_answer() -> None:
    prompt = build_prompt("Q?", ["a", "b"], cot=True)
    assert prompt.endswith(COT_TRIGGER)
    assert ANSWER_TRIGGER not in prompt


def test_options_block_letters_track_option_count() -> None:
    assert options_block(["one", "two", "three"]) == "A. one\nB. two\nC. three"
