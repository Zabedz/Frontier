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
    """Where a calibration corpus lives and how each row renders to a string.

    ``paragraphs`` joins that many consecutive prose lines of a ``text`` corpus into one
    sample, after blank and heading lines are dropped.
    """

    hf_path: str
    hf_config: str | None
    split: str
    render: Render
    paragraphs: int = 1


# WikiText-2 paragraphs run ~135 tokens against ~370 for an MMLU prompt, so three per sample
# keep the two corpora's calibration token budgets level.
PARAGRAPHS_PER_SAMPLE = 3


# auxiliary_train is disjoint from the test subset the ECE is computed on, so the
# calibration set cannot leak into the eval.
CALIBRATION_CORPORA: dict[CalibrationCorpus, CorpusSpec] = {
    "in_domain": CorpusSpec("cais/mmlu", "all", "auxiliary_train", "mcq"),
    "ood": CorpusSpec(
        "Salesforce/wikitext", "wikitext-2-raw-v1", "train", "text", PARAGRAPHS_PER_SAMPLE
    ),
}


def _render_row(render: Render, row: Any) -> str:
    if render == "mcq":
        return build_prompt(str(row["question"]), _as_options(row["choices"]))
    return str(row["text"])


def _prose_runs(dataset: Any, paragraphs: int) -> Any:
    """Drop blank and ``= heading =`` lines, then join runs of ``paragraphs`` lines.

    A trailing run shorter than ``paragraphs`` is dropped, so every sample is full length.
    """
    prose = [
        line.strip()
        for line in dataset["text"]
        if line.strip() and not line.strip().startswith("=")
    ]
    runs = [
        "\n\n".join(prose[start : start + paragraphs])
        for start in range(0, len(prose) - paragraphs + 1, paragraphs)
    ]
    return type(dataset).from_dict({"text": runs})


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
    if spec.render == "text":
        dataset = _prose_runs(dataset, spec.paragraphs)
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
