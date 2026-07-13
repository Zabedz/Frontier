"""Zero-shot and CoT prompt building. Pure strings, no tokeniser.

The non-CoT prompt ends in ``ANSWER_TRIGGER`` with no trailing space, so the answer
letter is the very next token (see the leading-space resolution in ``provider.py``).
The triggers are module constants so a test pins the exact text and a later config
can override them without touching the call sites.
"""

from __future__ import annotations

from collections.abc import Sequence

from frontier.eval.records import letters_for

INSTRUCTION = "Answer the following multiple choice question with the letter of the correct option."
COT_TRIGGER = "Let's think step by step."
ANSWER_TRIGGER = "Answer:"


def options_block(options: Sequence[str]) -> str:
    """Render options as ``"A. ...\\nB. ..."`` with one answer letter per option."""
    letters = letters_for(len(options))
    return "\n".join(f"{letter}. {text}" for letter, text in zip(letters, options, strict=True))


def build_prompt(question: str, options: Sequence[str], *, cot: bool = False) -> str:
    """Format the zero-shot MCQ prompt.

    The non-CoT form ends in ``ANSWER_TRIGGER`` so the next token is the answer
    letter::

        {INSTRUCTION}

        Question: {question}
        A. {opt0}
        B. {opt1}
        ...
        Answer:

    The CoT form ends in ``COT_TRIGGER`` instead and omits the answer trigger. The
    generation loop (the backend WP) samples the CoT, appends ``"\\n" +
    ANSWER_TRIGGER``, and reads the letter logits at that position. This builds the
    string and fixes where the letter is read; it does not generate.
    """
    header = f"{INSTRUCTION}\n\nQuestion: {question}\n{options_block(options)}"
    trigger = COT_TRIGGER if cot else ANSWER_TRIGGER
    return f"{header}\n{trigger}"
