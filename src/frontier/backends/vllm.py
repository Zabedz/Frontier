"""The vLLM Track-B logit provider.

vLLM does not expose raw logit tensors; it returns per-position ``logprobs``. The
candidate letters must be present in what comes back for every item and every option
order, so the provider requests ``logprobs=-1`` (the full vocab) and reads only the
known letter ids out of each returned dict. Under the V1 engine the returned logprobs
are the raw model output, taken before any logits processor and before temperature, so
an ``allowed_token_ids`` mask does not reshape the returned top-k; full-vocab is the
construction that guarantees every letter is present at any rank.

Why this reproduces the HF Track-A confidence exactly: vLLM returns
``logprob = logit - logsumexp(vocab)``, a single per-position constant subtracted from
every candidate's logit. The eval core softmaxes only the candidate-letter values, and
softmax is invariant to a shared additive constant, so the candidate distribution is
bit-for-bit the quantity the HF backend computes. Every call runs at ``temperature=1.0``,
which makes any temperature scaling a no-op wherever the backend applies it. The
FP16-on-vLLM fidelity gate is the empirical proof of this equivalence.

Prompt fidelity: the provider holds a HF ``AutoTokenizer``, reuses ``hf.chat_wrap`` to
build the exact Track-A prompt string, tokenises it with ``add_special_tokens=False``
(the chat template already carries the special tokens), and feeds the token ids straight
to ``generate``, bypassing vLLM's own chat templating so the forward pass matches HF.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from frontier.backends.hf import DEFAULT_REVISION, chat_wrap, resolve_candidates
from frontier.eval.records import FloatArray, IntArray, letters_for
from frontier.latency.native import vllm_gpu_memory_utilization

# vLLM's TokensPrompt is a TypedDict, so a plain dict with this key is exactly what
# ``generate`` accepts, and building it here avoids importing vllm on the CPU test path.
_PROMPT_TOKEN_IDS = "prompt_token_ids"


class VllmLogitProvider:
    """Answer-position candidate logprobs from a vLLM engine, HF-comparable at ``T=1``.

    ``model`` is the compressed-tensors checkpoint dir for a quantised variant, or the
    base ``model_id`` for the FP16 fidelity gate (vLLM serves the HF model directly).
    ``engine``, ``tokenizer``, and ``sampling_params_cls`` are injectable so the
    extraction is driven by a fake on CPU; on the pod they are ``None`` and the real
    ``vllm.LLM`` / ``AutoTokenizer`` / ``SamplingParams`` are built lazily on first use.
    """

    def __init__(
        self,
        *,
        model: str,
        tokenizer_id: str,
        device: str,
        weight_dtype: str,
        max_letters: int = 8,
        seed: int = 0,
        revision: str = DEFAULT_REVISION,
        engine: Any = None,
        tokenizer: Any = None,
        sampling_params_cls: Any = None,
    ) -> None:
        self.model = model
        self.device = device
        self.weight_dtype = weight_dtype
        self._tokenizer_id = tokenizer_id
        self._max_letters = max_letters
        self._seed = seed
        self._revision = revision
        self._engine = engine
        self._hf_tokenizer = tokenizer
        self._sampling_cls = sampling_params_cls
        self._sampling_params: Any = None
        self._vocab = 0
        self._letter_ids: IntArray = np.empty(0, dtype=np.intp)
        self._loaded = False
        self._backend_version = "unknown"
        self._candidate_cache: dict[tuple[str, ...], IntArray] = {}

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        if self._engine is None:
            self._build_engine()  # pragma: no cover
        self._sampling_params = self._sampling_cls(
            temperature=1.0, max_tokens=1, logprobs=-1, seed=self._seed
        )
        self._vocab = len(self._hf_tokenizer)
        self._letter_ids = resolve_candidates(self._hf_tokenizer, letters_for(self._max_letters))
        self._loaded = True

    def _build_engine(self) -> None:  # pragma: no cover
        import transformers  # noqa: PLC0415
        import vllm  # noqa: PLC0415

        self._hf_tokenizer = transformers.AutoTokenizer.from_pretrained(
            self._tokenizer_id, revision=self._revision
        )
        # An absolute budget rather than a fixed fraction: the engine's reservation is what
        # the latency probe records as this row's peak VRAM, and both must ask for the same
        # amount on any card the pod happens to hold.
        self._engine = vllm.LLM(
            model=self.model,
            dtype="auto",
            seed=self._seed,
            max_logprobs=-1,
            gpu_memory_utilization=vllm_gpu_memory_utilization(),
            enforce_eager=False,
        )
        self._sampling_cls = vllm.SamplingParams
        self._backend_version = str(vllm.__version__)

    def _token_prompt(self, prompt: str) -> dict[str, list[int]]:
        text = chat_wrap(self._hf_tokenizer, prompt)
        return {_PROMPT_TOKEN_IDS: self._hf_tokenizer.encode(text, add_special_tokens=False)}

    @property
    def backend_version(self) -> str:
        """The installed ``vllm`` version, or ``"unknown"`` before load."""
        return self._backend_version

    def candidate_token_ids(self, letters: Sequence[str]) -> IntArray:
        """One answer-position token id per letter, cached per letter tuple.

        Resolved through the same leading-space-then-bare fallback the HF backend uses,
        so the ids the eval core gathers are exactly the ids the full-vocab logprob row
        is filled at. Raises ``ValueError`` when more letters are requested than
        ``max_letters``: ``next_token_logits`` fills the row only at the ``A``..letter-N
        ids resolved at load, so a letter beyond that would silently read ``-inf``
        (probability 0) instead of its real logprob.
        """
        self._ensure_loaded()
        if len(letters) > self._max_letters:
            last = chr(ord("A") + self._max_letters - 1)
            raise ValueError(
                f"vLLM provider resolves {self._max_letters} answer letters (A-{last}) but "
                f"{len(letters)} were requested; raise max_letters for a wider task"
            )
        key = tuple(letters)
        cached = self._candidate_cache.get(key)
        if cached is not None:
            return cached
        ids = resolve_candidates(self._hf_tokenizer, letters)
        self._candidate_cache[key] = ids
        return ids

    def next_token_logits(self, prompts: Sequence[str]) -> FloatArray:
        """Answer-position candidate logprobs for a batch of chat-wrapped prompts.

        Each prompt is chat-wrapped and tokenised to match the HF forward pass, generated
        for one token with full-vocab logprobs, and read into a ``(len(prompts), vocab)``
        row of ``-inf`` at only the resolved letter ids. The eval core gathers the item's
        candidate ids out of that row and softmaxes, so the ``-inf`` non-letter entries
        are never touched.
        """
        self._ensure_loaded()
        token_prompts = [self._token_prompt(prompt) for prompt in prompts]
        outputs = self._engine.generate(token_prompts, self._sampling_params)
        rows: FloatArray = np.full((len(prompts), self._vocab), -np.inf, dtype=np.float64)
        for i, output in enumerate(outputs):
            logprobs = output.outputs[0].logprobs[0]
            for token_id in self._letter_ids:
                entry = logprobs.get(int(token_id))
                if entry is not None:
                    rows[i, int(token_id)] = float(entry.logprob)
        return rows
