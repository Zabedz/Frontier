"""QAT reads the shared corpus and schedule code: one implementation, re-exported."""

from __future__ import annotations

from frontier.quantize import qat
from frontier.training import corpus, loop


def test_qat_reexports_the_shared_corpus_and_schedule() -> None:
    assert qat.pack_tokens is corpus.pack_tokens
    assert qat.stream_corpus is corpus.stream_corpus
    assert qat.TRAINING_CORPORA is corpus.TRAINING_CORPORA
    assert qat.lr_multiplier is loop.lr_multiplier
