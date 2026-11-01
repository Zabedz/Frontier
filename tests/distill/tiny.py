"""Offline fixtures for the distillation tests: a word tokenizer, a random tiny Llama, a
successor teacher, and cache builders over them. Nothing here touches the network.

Plain functions, as in ``tests/io/rows.py``: a second ``conftest.py`` would collide with
``tests/metrics/conftest.py`` as a top-level module under mypy.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

from frontier.distill.cache import (
    CacheMeta,
    CacheRequest,
    Forward,
    TeacherFacts,
    fill_targets,
    finish_cache,
    start_cache,
)
from frontier.pipeline.config import resolve_config
from frontier.schema import VariantConfig

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"
ARMS = ("distill-softlabel-topk", "distill-corpus", "distill-hardlabel")
WORDS = 60
EOS_ID = 1
# Above the tokenizer's 62 entries, as Qwen's 151,936 rows exceed its tokenizer, and above
# the shipped top_k of 64.
TINY_VOCAB = 96
SUCCESSOR_BOOST = 8.0
NOISE_SCALE = 0.5
ONE_STEP_TOKENS = 16_384  # one optimiser step at 16 rows of 1024
SHORT_ROW = 8


def word_tokenizer(words: int = WORDS) -> Any:
    """WordLevel over ``<unk>``, ``<eos>``, ``w0..w<words-1>``, split on whitespace; eos id 1."""
    tokenizers = pytest.importorskip("tokenizers")
    transformers = pytest.importorskip("transformers")
    vocab = {"<unk>": 0, "<eos>": EOS_ID, **{f"w{i}": i + 2 for i in range(words)}}
    model = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab=vocab, unk_token="<unk>"))
    model.pre_tokenizer = tokenizers.pre_tokenizers.Whitespace()
    return transformers.PreTrainedTokenizerFast(
        tokenizer_object=model, eos_token="<eos>", unk_token="<unk>"
    )


def tiny_llama_dir(
    root: Path,
    *,
    name: str = "tiny-llama",
    words: int = WORDS,
    vocab: int = TINY_VOCAB,
    seed: int = 0,
) -> Path:
    """A random two-layer Llama with tied embeddings, saved with the word tokenizer."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    config = transformers.LlamaConfig(
        vocab_size=vocab,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        max_position_embeddings=1024,
        tie_word_embeddings=True,
        eos_token_id=EOS_ID,
    )
    torch.manual_seed(seed)
    path = root / name
    transformers.LlamaForCausalLM(config).save_pretrained(path)
    word_tokenizer(words).save_pretrained(path)
    return path


def load_tiny(path: Path) -> Any:
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    return transformers.AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32)


def word_texts(documents: int = 400, length: int = 60, *, seed: int = 0) -> list[str]:
    """Documents of random words, enough for 20 rows of 1024 tokens with the EOS joins."""
    rng = np.random.default_rng(seed)
    return [
        " ".join(f"w{index}" for index in rng.integers(0, WORDS, size=length))
        for _ in range(documents)
    ]


def tiny_variant(
    model_dir: Path, *, name: str = "distill-softlabel-topk", student_dir: Path | None = None
) -> VariantConfig:
    """A shipped distill config, smoke-resolved, with teacher and student at ``model_dir``."""
    resolved = resolve_config(
        CONFIG_ROOT / "variants" / f"{name}.yaml", mode="smoke", config_root=CONFIG_ROOT
    )
    model = replace(
        resolved.variant.model,
        model_id=str(model_dir),
        student_id=str(student_dir or model_dir),
    )
    return replace(resolved.variant, model=model)


def successor_forward(vocab: int = TINY_VOCAB, *, seed: int = 0) -> Forward:
    """Logits ``+8`` at ``(x_t + 1) mod V`` over a fixed random row picked by ``x_t``.

    On arithmetic rows, ``x_t = (x_0 + t) mod V``, the top-1 at ``t`` is ``x_{t+1}``.
    """
    torch = pytest.importorskip("torch")
    generator = torch.Generator().manual_seed(seed)
    table = NOISE_SCALE * torch.randn(vocab, vocab, generator=generator)
    table += SUCCESSOR_BOOST * torch.nn.functional.one_hot((torch.arange(vocab) + 1) % vocab, vocab)

    def forward(ids: Any) -> Any:
        return table[ids]

    return forward


def arithmetic_rows(
    n_rows: int, seq_len: int = SHORT_ROW, vocab: int = TINY_VOCAB
) -> npt.NDArray[np.int64]:
    """Row ``r`` counts up from ``7 r`` modulo ``vocab``."""
    starts = 7 * np.arange(n_rows, dtype=np.int64)[:, None]
    rows: npt.NDArray[np.int64] = (starts + np.arange(seq_len, dtype=np.int64)) % vocab
    return rows


def request_for(rows: npt.NDArray[np.int64], *, top_k: int, heldout_rows: int = 0) -> CacheRequest:
    n_rows, seq_len = rows.shape
    return CacheRequest(
        teacher_id="successor",
        teacher_revision="main",
        corpus="fineweb_edu",
        seq_len=seq_len,
        top_k=top_k,
        train_rows=n_rows - heldout_rows,
        heldout_rows=heldout_rows,
        budget_tokens=(n_rows - heldout_rows) * seq_len,
        capped=False,
    )


def cpu_facts(vocab: int = TINY_VOCAB) -> TeacherFacts:
    return TeacherFacts(
        vocab_size=vocab,
        eos_token_id=EOS_ID,
        tokenizer_fingerprint="successor",
        commit=None,
        dtype="float32",
        device="cpu",
        deterministic=False,
        cublas_workspace_config=None,
    )


def build_cache(
    out: Path,
    rows: npt.NDArray[np.int64],
    forward: Forward,
    *,
    top_k: int = 3,
    heldout_rows: int = 0,
    batch_rows: int = 2,
    flush_rows: int = 2,
) -> CacheMeta:
    """Start, fill, and finish a cache with no fidelity check."""
    facts = cpu_facts()
    start_cache(
        out,
        rows,
        request_for(rows, top_k=top_k, heldout_rows=heldout_rows),
        facts,
        batch_rows=batch_rows,
        flush_rows=flush_rows,
    )
    fill_targets(out, forward, facts=facts, device="cpu")
    return finish_cache(out)
