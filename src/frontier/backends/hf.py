"""The Hugging Face Track-A logit backend.

``HFLogitProvider`` structurally satisfies the WP2 ``LogitProvider`` seam: it returns
next-token logits at the answer position for a batch of prompts and resolves each
answer letter to a single token id. Loading is lazy, so importing this module costs
nothing and does not need the ``hf`` dependency group; ``transformers`` and ``torch``
are imported the first time the model is used.

The prompt seam is the load-bearing part. WP2's ``build_prompt`` returns plain
user-turn text ending in ``Answer:``; an instruct model reads that letter honestly
only inside its own chat format, so ``chat_wrap`` splits the prompt on
``"\\n" + ANSWER_TRIGGER``, feeds the question through the tokenizer chat template with
``add_generation_prompt=True``, and re-appends ``Answer:`` as an assistant prefill.
The next token after ``Answer:`` is then ``" A"``, which is what the WP2
leading-space candidate resolution expects. Raw concatenation (feed ``build_prompt``
straight through, treating the instruct model as a base LM) is the documented
fallback; it is a one-line change here.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np

from frontier.eval.prompts import ANSWER_TRIGGER
from frontier.eval.provider import Tokenizer, resolve_candidate_ids
from frontier.eval.records import FloatArray, IntArray

DEFAULT_REVISION = "main"

# Configured weight_dtype -> torch dtype name, for the CUDA compute path only. On CPU
# the compute dtype is always float32 (fp16/bf16 matmul on CPU is unsupported or very
# slow), so weight_dtype does not reach this map there.
_CUDA_COMPUTE_DTYPE = {
    "fp16": "float16",
    "bf16": "bfloat16",
    "fp32": "float32",
    "float16": "float16",
    "bfloat16": "bfloat16",
    "float32": "float32",
}


class _ChatTokenizer(Protocol):
    """The one chat-template call ``chat_wrap`` needs."""

    def apply_chat_template(
        self,
        conversation: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str: ...


def resolve_device(mode: str, *, cuda_available: bool | None = None) -> str:
    """The device string for a run mode.

    ``smoke`` is always ``"cpu"``. ``full`` is ``"cuda"`` when a GPU is present, else
    ``"cpu"``. ``cuda_available`` is injectable so a test pins the branch without a
    GPU; it defaults to ``torch.cuda.is_available()`` (imported lazily).
    """
    if mode == "smoke":
        return "cpu"
    if cuda_available is None:
        import torch  # noqa: PLC0415

        cuda_available = bool(torch.cuda.is_available())
    return "cuda" if cuda_available else "cpu"


def resolve_dtype(device: str, weight_dtype: str) -> str:
    """The torch dtype name actually used for the forward pass.

    On ``cpu`` the compute dtype is ``"float32"`` regardless of the configured
    ``weight_dtype``, because fp16/bf16 matmul on CPU is unsupported or very slow and
    the smoke path is a wiring proof, not a real dtype measurement. On ``cuda`` the
    configured ``weight_dtype`` maps to its torch dtype. The result row still records
    the configured ``weight_dtype``, so it describes the variant, not the CPU
    fallback; the device recorded in ``hardware_id`` is where the split is visible.
    """
    if device == "cpu":
        return "float32"
    try:
        return _CUDA_COMPUTE_DTYPE[weight_dtype]
    except KeyError:
        raise ValueError(
            f"no torch compute dtype for weight_dtype {weight_dtype!r} on device {device!r}"
        ) from None


def chat_wrap(tokenizer: _ChatTokenizer, prompt: str) -> str:
    """Wrap a non-CoT ``build_prompt`` string in the model's chat template.

    The prompt must end in ``"\\n" + ANSWER_TRIGGER``; the part before is the user
    content and ``Answer:`` becomes the assistant prefill re-appended after
    ``add_generation_prompt=True``. A prompt that does not end in the trigger (the CoT
    case, out of WP3 scope) raises ``ValueError`` so the non-CoT contract is enforced
    rather than silently mis-wrapped. ``continue_final_message=True`` on an assistant
    message is a version-fragile alternative to this string append; the append is
    simpler and stable across transformers releases.
    """
    suffix = f"\n{ANSWER_TRIGGER}"
    if not prompt.endswith(suffix):
        raise ValueError(
            f"non-CoT prompt must end in {suffix!r} to chat-wrap, got tail {prompt[-32:]!r}"
        )
    header = prompt[: -len(suffix)]
    wrapped = tokenizer.apply_chat_template(
        [{"role": "user", "content": header}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return wrapped + ANSWER_TRIGGER


def resolve_candidates(tokenizer: Tokenizer, letters: Sequence[str]) -> IntArray:
    """Resolve the answer letters to single token ids, with the no-space fallback.

    Tries the ``" A"`` leading-space form the WP2 core assumes; if a letter splits it
    retries the bare-letter form. If both split, the ``ValueError`` from the bare form
    propagates: the tokenizer genuinely cannot represent the letter as one token.
    """
    try:
        return resolve_candidate_ids(tokenizer, letters, leading_space=True)
    except ValueError:
        return resolve_candidate_ids(tokenizer, letters, leading_space=False)


class HFLogitProvider:
    """Next-token answer-position logits from a Hugging Face causal LM.

    Lazy: the tokenizer and model load on first use, left-padded so the last column of
    every batched row is the real final token. The provider structurally satisfies the
    WP2 ``LogitProvider`` protocol and carries ``backend_version`` so the runner can
    stamp it without widening that protocol.
    """

    def __init__(
        self,
        *,
        model_id: str,
        device: str,
        weight_dtype: str,
        revision: str = DEFAULT_REVISION,
    ) -> None:
        self.model_id = model_id
        self.device = device
        self.weight_dtype = weight_dtype
        self.revision = revision
        self._tokenizer: Any = None
        self._model: Any = None
        self._backend_version = "unknown"
        self._candidate_cache: dict[tuple[str, ...], IntArray] = {}

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        import torch  # noqa: PLC0415
        import transformers  # noqa: PLC0415

        tokenizer = transformers.AutoTokenizer.from_pretrained(
            self.model_id, revision=self.revision
        )
        tokenizer.padding_side = "left"
        torch_dtype = getattr(torch, resolve_dtype(self.device, self.weight_dtype))
        model = transformers.AutoModelForCausalLM.from_pretrained(
            self.model_id, revision=self.revision, dtype=torch_dtype
        )
        model.to(self.device)
        model.eval()
        self._tokenizer = tokenizer
        self._model = model
        self._backend_version = transformers.__version__

    @property
    def backend_version(self) -> str:
        """The installed ``transformers`` version, or ``"unknown"`` before load."""
        return self._backend_version

    def loaded_model(self) -> tuple[Any, Any]:
        """Ensure the model is loaded and return ``(model, tokenizer)`` for the rig.

        The latency rig drives timed generation on these same weights, so the model
        loads once for both scoring and timing rather than a second time.
        """
        self._ensure_loaded()
        return self._model, self._tokenizer

    def candidate_token_ids(self, letters: Sequence[str]) -> IntArray:
        """One answer-position token id per letter, cached per letter tuple."""
        self._ensure_loaded()
        key = tuple(letters)
        cached = self._candidate_cache.get(key)
        if cached is not None:
            return cached
        ids = resolve_candidates(self._tokenizer, letters)
        self._candidate_cache[key] = ids
        return ids

    def next_token_logits(self, prompts: Sequence[str]) -> FloatArray:
        """Answer-position logits for a batch of chat-wrapped prompts.

        Each prompt is chat-wrapped, the batch is left-padded and tokenized with the
        template's special tokens already in place (``add_special_tokens=False`` avoids
        a doubled BOS), and one forward pass yields ``logits[:, -1, :]`` as a
        ``(len(prompts), vocab)`` float64 array.
        """
        self._ensure_loaded()
        import torch  # noqa: PLC0415

        texts = [chat_wrap(self._tokenizer, prompt) for prompt in prompts]
        encoded = self._tokenizer(
            texts, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(self.device)
        with torch.inference_mode():
            output = self._model(**encoded)
        final = output.logits[:, -1, :].to(torch.float32).cpu().numpy()
        return np.asarray(final, dtype=np.float64)
