"""The training corpus, streamed in dataset order and packed into fixed-length token rows;
the QAT and distillation arms both read it through here."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator, Mapping
from typing import Protocol

import numpy as np
import numpy.typing as npt

from frontier.schema import TrainingCorpus

# (hf path, config, split, text field). Streamed in the dataset's own order, so every seed
# trains on the same tokens and the seed only reorders them.
TRAINING_CORPORA: dict[TrainingCorpus, tuple[str, str, str, str]] = {
    "fineweb_edu": ("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", "text"),
}

TextStream = Callable[[TrainingCorpus], Iterable[str]]
TokenRows = npt.NDArray[np.int64]  # (n_rows, seq_len) packed token ids


class Encoder(Protocol):
    """The tokenizer call ``pack_tokens`` makes."""

    eos_token_id: int | None

    def __call__(self, text: str, *, add_special_tokens: bool) -> Mapping[str, list[int]]: ...


def pack_tokens(
    texts: Iterable[str], tokenizer: Encoder, *, seq_len: int, max_tokens: int
) -> TokenRows:
    """Tokenise, join with EOS, and cut into full ``seq_len`` rows until ``max_tokens``.

    Returns an ``int64`` array ``(n_rows, seq_len)``; a trailing partial row is dropped.
    Raises ``ValueError`` when the stream ends before one full row.
    """
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer has no eos_token_id to separate documents")
    n_rows = max(max_tokens // seq_len, 1)
    needed = n_rows * seq_len
    buffer: list[int] = []
    for text in texts:
        buffer.extend(tokenizer(text, add_special_tokens=False)["input_ids"])
        buffer.append(tokenizer.eos_token_id)
        if len(buffer) >= needed:
            break
    usable = (min(len(buffer), needed) // seq_len) * seq_len
    if usable == 0:
        raise ValueError(f"text stream ended before one {seq_len}-token row")
    return np.asarray(buffer[:usable], dtype=np.int64).reshape(-1, seq_len)


def stream_corpus(corpus: TrainingCorpus) -> Iterator[str]:  # pragma: no cover
    """The corpus's documents in the dataset's streamed order."""
    import datasets  # noqa: PLC0415

    hf_path, config, split, field = TRAINING_CORPORA[corpus]
    for row in datasets.load_dataset(hf_path, config, split=split, streaming=True):
        yield str(row[field])
