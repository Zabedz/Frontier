"""The llama.cpp Track-B logit provider: true full-vocab logits read at the last position
of ``scores``, with the prompt chat-wrapped through the HF tokenizer to match Track A.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from frontier.backends.hf import DEFAULT_REVISION, chat_wrap
from frontier.eval.provider import SINGLE_TOKEN
from frontier.eval.records import FloatArray, IntArray

DEFAULT_N_CTX = 4096


class LlamaCppLogitProvider:
    """Answer-position full-vocab logits from a llama.cpp GGUF model.

    ``llama`` and ``tokenizer`` are injectable so the extraction runs against fakes on CPU;
    left ``None``, the real ones are built on first use.
    """

    def __init__(
        self,
        *,
        gguf_path: Path,
        tokenizer_id: str,
        device: str,
        n_gpu_layers: int,
        weight_dtype: str,
        n_ctx: int = DEFAULT_N_CTX,
        seed: int = 0,
        revision: str = DEFAULT_REVISION,
        llama: Any = None,
        tokenizer: Any = None,
    ) -> None:
        self.gguf_path = gguf_path
        self.device = device
        self.weight_dtype = weight_dtype
        self._tokenizer_id = tokenizer_id
        self._n_gpu_layers = n_gpu_layers
        self._n_ctx = n_ctx
        self._seed = seed
        self._revision = revision
        self._llama = llama
        self._hf_tokenizer = tokenizer
        self._backend_version = "unknown"
        self._candidate_cache: dict[tuple[str, ...], IntArray] = {}

    def _ensure_loaded(self) -> None:
        if self._llama is not None:
            return
        self._build_llama()  # pragma: no cover

    def _build_llama(self) -> None:  # pragma: no cover
        # torch first binds the venv's nccl 2.28.9; otherwise llama_cpp's CUDA libraries pull
        # the base image's 2.25.1 and torch fails on ncclCommShrink. I001 protects the order.
        import torch  # noqa: F401, I001, PLC0415
        import llama_cpp  # noqa: PLC0415
        import transformers  # noqa: PLC0415

        self._llama = llama_cpp.Llama(
            model_path=str(self.gguf_path),
            n_gpu_layers=self._n_gpu_layers,
            n_ctx=self._n_ctx,
            # Without logits_all, scores is (n_batch, vocab) and the last-position read
            # runs past it for any multi-batch prompt.
            logits_all=True,
            seed=self._seed,
            verbose=False,
        )
        self._hf_tokenizer = transformers.AutoTokenizer.from_pretrained(
            self._tokenizer_id, revision=self._revision
        )
        self._backend_version = str(llama_cpp.__version__)

    @property
    def backend_version(self) -> str:
        """The installed ``llama_cpp`` version, or ``"unknown"`` before load."""
        return self._backend_version

    def candidate_token_ids(self, letters: Sequence[str]) -> IntArray:
        """One answer-position token id per letter, cached per letter tuple.

        Resolution runs through the llama tokeniser, so the ids index the same vocab the
        ``scores`` row is over.
        """
        self._ensure_loaded()
        key = tuple(letters)
        cached = self._candidate_cache.get(key)
        if cached is not None:
            return cached
        ids = np.asarray([self._resolve_letter(letter) for letter in letters], dtype=np.intp)
        self._candidate_cache[key] = ids
        return ids

    def _resolve_letter(self, letter: str) -> int:
        spaced = self._llama.tokenize(f" {letter}".encode(), add_bos=False, special=False)
        if len(spaced) == SINGLE_TOKEN:
            return int(spaced[0])
        bare = self._llama.tokenize(letter.encode(), add_bos=False, special=False)
        if len(bare) == SINGLE_TOKEN:
            return int(bare[0])
        raise ValueError(
            f"answer letter {letter!r} is not a single llama token: spaced={spaced}, bare={bare}"
        )

    def next_token_logits(self, prompts: Sequence[str]) -> FloatArray:
        """Answer-position full-vocab logits for a batch of prompts.

        ``reset()`` runs per prompt; a context carried over would put the last-position read
        at the wrong offset.
        """
        self._ensure_loaded()
        rows = []
        for prompt in prompts:
            text = chat_wrap(self._hf_tokenizer, prompt)
            # The chat template already emits Qwen's specials as text, and Qwen2.5 has no BOS.
            tokens = self._llama.tokenize(text.encode("utf-8"), add_bos=False, special=True)
            self._llama.reset()
            self._llama.eval(tokens)
            rows.append(np.asarray(self._llama.scores[self._llama.n_tokens - 1], dtype=np.float64))
        return np.stack(rows)
