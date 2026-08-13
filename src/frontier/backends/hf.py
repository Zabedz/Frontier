"""The Hugging Face Track-A logit backend. ``transformers`` and ``torch`` are imported on
first use, so importing this module needs neither.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Protocol

import numpy as np

from frontier.backends.bnb import bnb_config_kwargs, is_bnb_dtype, resolve_bnb_compute_dtype
from frontier.eval.prompts import ANSWER_TRIGGER
from frontier.eval.provider import Tokenizer, resolve_candidate_ids
from frontier.eval.records import FloatArray, IntArray

DEFAULT_REVISION = "main"

# Configured weight_dtype -> torch dtype name, for the CUDA compute path only.
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
    """The device string for a run mode: ``smoke`` is always CPU, ``full`` follows the GPU.

    ``cuda_available`` is injectable so a test pins the branch on a machine without a GPU.
    """
    if mode == "smoke":
        return "cpu"
    if cuda_available is None:
        import torch  # noqa: PLC0415

        cuda_available = bool(torch.cuda.is_available())
    return "cuda" if cuda_available else "cpu"


def resolve_dtype(device: str, weight_dtype: str) -> str:
    """The torch dtype name actually used for the forward pass.

    ``cpu`` is always ``"float32"``, since fp16/bf16 matmul there is unsupported or very slow.
    The result row still records the configured ``weight_dtype``, describing the variant, and
    ``hardware_id`` carries the device the run actually used.
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

    An instruct model reads the answer letter honestly only inside its own chat format. The
    question goes through the template and ``Answer:`` is re-appended as an assistant prefill,
    so the next token is ``" A"`` and the leading-space candidate resolution holds.
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
    # continue_final_message is the API alternative; it shifts across transformers releases.
    return wrapped + ANSWER_TRIGGER


def resolve_candidates(tokenizer: Tokenizer, letters: Sequence[str]) -> IntArray:
    """Resolve the answer letters to single token ids.

    The ``" A"`` leading-space form the eval core assumes is tried first, then the bare
    letter; when both split, the tokenizer cannot represent the letter as one token.
    """
    try:
        return resolve_candidate_ids(tokenizer, letters, leading_space=True)
    except ValueError:
        return resolve_candidate_ids(tokenizer, letters, leading_space=False)


class HFLogitProvider:
    """Next-token answer-position logits from a Hugging Face causal LM.

    The tokenizer and model load on first use, left-padded so the last column of every
    batched row is that row's real final token.
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
        if self.device != "cpu" and is_bnb_dtype(self.weight_dtype):
            model = self._load_bnb(transformers, torch)  # pragma: no cover
        else:
            torch_dtype = getattr(torch, resolve_dtype(self.device, self.weight_dtype))
            model = transformers.AutoModelForCausalLM.from_pretrained(
                self.model_id, revision=self.revision, dtype=torch_dtype
            )
            model.to(self.device)
        model.eval()
        self._tokenizer = tokenizer
        self._model = model
        self._backend_version = transformers.__version__

    def _load_bnb(self, transformers: Any, torch: Any) -> Any:  # pragma: no cover
        """Load the model quantised through bitsandbytes on the CUDA path.

        A bnb model is placed by ``device_map`` at load and cannot be moved with ``.to``
        afterwards, so the caller skips its ``.to`` for this branch.
        """
        compute_name = resolve_bnb_compute_dtype(self.weight_dtype)
        kwargs = bnb_config_kwargs(self.weight_dtype, compute_dtype=compute_name)
        if "bnb_4bit_compute_dtype" in kwargs:
            kwargs["bnb_4bit_compute_dtype"] = getattr(torch, compute_name)
        quant_config = transformers.BitsAndBytesConfig(**kwargs)
        return transformers.AutoModelForCausalLM.from_pretrained(
            self.model_id,
            revision=self.revision,
            quantization_config=quant_config,
            dtype=getattr(torch, compute_name),
            device_map={"": 0},
        )

    @property
    def backend_version(self) -> str:
        """The installed ``transformers`` version, or ``"unknown"`` before load."""
        return self._backend_version

    def loaded_model(self) -> tuple[Any, Any]:
        """Load if needed and return ``(model, tokenizer)`` so the rig times the scored weights."""
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
        """Answer-position logits for a batch of prompts, shape ``(len(prompts), vocab)``.

        ``add_special_tokens=False`` avoids a doubled BOS: the chat template already emitted
        the special tokens as text.
        """
        self._ensure_loaded()
        import torch  # noqa: PLC0415

        texts = [chat_wrap(self._tokenizer, prompt) for prompt in prompts]
        encoded = self._tokenizer(
            texts, return_tensors="pt", padding=True, add_special_tokens=False
        ).to(self.device)
        with torch.inference_mode():
            # Only the answer position is ever read, and the vocab is ~152k wide.
            output = self._model(**encoded, logits_to_keep=1)
        final = output.logits[:, -1, :].to(torch.float32).cpu().numpy()
        return np.asarray(final, dtype=np.float64)
