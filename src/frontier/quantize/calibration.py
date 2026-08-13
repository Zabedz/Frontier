"""The calibration-set builder for the compressed-tensors producers.

GPTQ, AWQ, and SmoothQuant all pick scales from a small calibration corpus, so the corpus
is a parameter: the out-of-domain ``CorpusSpec`` arrives at matched sample count, seqlen,
and sampling seed, leaving the corpus as the only difference along that axis. The dataset
loader is injectable, so render-and-tokenize is unit-tested on a tiny in-memory
``datasets.Dataset`` with a fake tokenizer.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from frontier.eval.prompts import build_prompt
from frontier.schema import CalibrationCorpus

Render = Literal["mcq", "text"]
DatasetLoader = Callable[[str, str | None, str], Any]


@dataclass(frozen=True, slots=True)
class CorpusSpec:
    """Where a calibration corpus lives and how each row renders to a string."""

    hf_path: str
    hf_config: str | None
    split: str
    render: Render


# auxiliary_train is disjoint from the test subset the ECE is computed on, so the
# calibration set cannot leak into the eval.
CALIBRATION_CORPORA: dict[CalibrationCorpus, CorpusSpec] = {
    "in_domain": CorpusSpec("cais/mmlu", "all", "auxiliary_train", "mcq"),
}


def _render_row(render: Render, row: Any) -> str:
    if render == "mcq":
        return build_prompt(str(row["question"]), _as_options(row["choices"]))
    return str(row["text"])


def _as_options(choices: Any) -> Sequence[str]:
    return [str(choice) for choice in choices]


def _load_dataset(hf_path: str, hf_config: str | None, split: str) -> Any:  # pragma: no cover
    import datasets  # noqa: PLC0415

    return datasets.load_dataset(hf_path, hf_config, split=split)


def build_calibration_dataset(
    corpus: CalibrationCorpus,
    tokenizer: Any,
    *,
    num_samples: int,
    max_seq_length: int,
    seed: int,
    loader: DatasetLoader | None = None,
) -> Any:
    """Load, render, shuffle(seed), select(num_samples), and tokenize a calibration set.

    ``mcq`` render puts each row through ``eval.prompts.build_prompt``, so the calibration
    activations sit in the same distribution as the MMLU eval. The tokenised output is
    unpadded and carries no text columns, the shape llm-compressor's ``oneshot`` expects.
    """
    try:
        spec = CALIBRATION_CORPORA[corpus]
    except KeyError:
        raise ValueError(
            f"calibration corpus {corpus!r} is not wired; available: {sorted(CALIBRATION_CORPORA)}"
        ) from None
    load = loader or _load_dataset
    dataset = load(spec.hf_path, spec.hf_config, spec.split)
    dataset = dataset.shuffle(seed=seed).select(range(num_samples))
    rendered = dataset.map(lambda row: {"text": _render_row(spec.render, row)})
    return rendered.map(
        lambda row: tokenizer(
            row["text"],
            padding=False,
            truncation=True,
            max_length=max_seq_length,
            add_special_tokens=False,
        ),
        remove_columns=rendered.column_names,
    )
