"""QAT producer: torchao fake-quant training, optionally through LoRA, saved as a plain HF
checkpoint.

torchao's QAT ``convert`` step on the trained model gives bit-identical logits to loading
the saved weights and quantising on load, so ``TorchaoLogitProvider`` serves QAT and the
PTQ control through one path and the two arms differ only in their weights. ``torch``,
``transformers``, ``peft``, and ``torchao`` are imported on first use.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable, Iterable, Iterator, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from frontier.backends.torchao import intx_weight_only_config, quantizes
from frontier.quantize.methods import QAT_LORA_METHOD, is_qat_method
from frontier.quantize.paths import checkpoint_path
from frontier.schema import QATSpec, TrainingCorpus, VariantConfig

# (hf path, config, split, text field). Streamed in the dataset's own order, so every seed
# trains on the same tokens and the seed only reorders them.
TRAINING_CORPORA: dict[TrainingCorpus, tuple[str, str, str, str]] = {
    "fineweb_edu": ("HuggingFaceFW/fineweb-edu", "sample-10BT", "train", "text"),
}

# The projections Qwen2 and Llama-family blocks carry; LoRA wraps every one.
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
GRAD_CLIP_NORM = 1.0
ADAM_BETAS = (0.9, 0.95)
LOG_EVERY_STEPS = 10
METADATA_FILE = "qat.json"
_COMPLETION_MARKERS = ("config.json", METADATA_FILE)

TextStream = Callable[[TrainingCorpus], Iterable[str]]
TokenRows = npt.NDArray[np.int64]  # (n_rows, seq_len) packed token ids


class _Encoder(Protocol):
    eos_token_id: int | None

    def __call__(self, text: str, *, add_special_tokens: bool) -> Mapping[str, list[int]]: ...


def pack_tokens(
    texts: Iterable[str], tokenizer: _Encoder, *, seq_len: int, max_tokens: int
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


def lr_multiplier(step: int, *, total_steps: int, warmup_steps: int) -> float:
    """Linear warmup to 1, then cosine decay to 0 at ``total_steps``."""
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def require_qat_spec(variant: VariantConfig) -> QATSpec:
    """The variant's ``qat`` block, checked against its quant method.

    Raises ``ValueError`` for a missing block, a LoRA method with no rank, or a
    full-parameter method that sets one.
    """
    if variant.quant is None or not is_qat_method(variant.quant.method):
        raise ValueError(f"variant {variant.name!r} is not a torchao QAT variant")
    spec = variant.qat
    if spec is None:
        raise ValueError(f"QAT variant {variant.name!r} has no qat block")
    lora = variant.quant.method == QAT_LORA_METHOD
    if lora and spec.lora_rank <= 0:
        raise ValueError(f"{variant.name!r} uses {QAT_LORA_METHOD!r} but sets no lora_rank")
    if not lora and spec.lora_rank:
        raise ValueError(f"{variant.name!r} trains every parameter but sets lora_rank")
    return spec


def produce_qat(
    variant: VariantConfig,
    backend: Mapping[str, Any],
    *,
    checkpoints_root: Path,
    seed: int,
    device: str,
    max_tokens: int | None = None,
    texts: TextStream | None = None,
) -> Path:
    """Train ``variant`` for ``seed`` and save it at ``checkpoint_path``; idempotent.

    ``max_tokens`` caps the configured budget (the smoke path). ``texts`` replaces the
    streamed corpus, for tests.
    """
    spec = require_qat_spec(variant)
    out = checkpoint_path(variant, backend, root=checkpoints_root, seed=seed)
    if out.is_dir() and all((out / marker).exists() for marker in _COMPLETION_MARKERS):
        return out
    budget = spec.train_tokens if max_tokens is None else min(max_tokens, spec.train_tokens)
    stream = (texts or stream_corpus)(spec.corpus)
    return _train(variant, spec, out, seed=seed, device=device, budget=budget, texts=stream)


def stream_corpus(corpus: TrainingCorpus) -> Iterator[str]:  # pragma: no cover
    """The corpus's documents in the dataset's streamed order."""
    import datasets  # noqa: PLC0415

    hf_path, config, split, field = TRAINING_CORPORA[corpus]
    for row in datasets.load_dataset(hf_path, config, split=split, streaming=True):
        yield str(row[field])


def _train(  # pragma: no cover
    variant: VariantConfig,
    spec: QATSpec,
    out: Path,
    *,
    seed: int,
    device: str,
    budget: int,
    texts: Iterable[str],
) -> Path:
    import torch  # noqa: PLC0415
    import transformers  # noqa: PLC0415

    assert variant.quant is not None
    lora = variant.quant.method == QAT_LORA_METHOD
    torch.manual_seed(seed)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        variant.model.model_id, revision=variant.model.model_revision
    )
    model = _prepared_model(variant, spec, device=device, lora=lora)
    rows = pack_tokens(texts, tokenizer, seq_len=spec.seq_len, max_tokens=budget)
    # An abandoned streamed parquet read blocks pyarrow's thread-pool shutdown, so the
    # process would hang at exit; closing the generator ends the read.
    close = getattr(texts, "close", None)
    if close is not None:
        close()
    order = np.random.default_rng(seed).permutation(len(rows))
    start = time.time()
    steps, final_loss = _run_steps(model, rows[order], spec, device=device)

    if lora:
        model = model.merge_and_unload()
    _drop_fake_quant(model)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    metadata = {
        "variant": variant.name,
        "seed": seed,
        "qat": asdict(spec),
        "tokens_trained": steps * spec.micro_batch_size * spec.grad_accum * spec.seq_len,
        "steps": steps,
        "final_loss": final_loss,
        "wall_seconds": round(time.time() - start, 1),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
    }
    (out / METADATA_FILE).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return out


def _prepared_model(  # pragma: no cover
    variant: VariantConfig, spec: QATSpec, *, device: str, lora: bool
) -> Any:
    """The base model with fake-quantised linears, LoRA-wrapped when ``lora``."""
    import torch  # noqa: PLC0415
    import transformers  # noqa: PLC0415
    from torchao.quantization import quantize_  # noqa: PLC0415
    from torchao.quantization.qat import QATConfig  # noqa: PLC0415

    assert variant.quant is not None
    cuda = device != "cpu"
    # Frozen LoRA bases train in bf16; full-parameter QAT keeps fp32 master weights.
    load_dtype = torch.bfloat16 if cuda and lora else torch.float32
    model = transformers.AutoModelForCausalLM.from_pretrained(
        variant.model.model_id, revision=variant.model.model_revision, dtype=load_dtype
    ).to(device)
    model.config.use_cache = False
    base = intx_weight_only_config(variant.quant.bit_width, variant.quant.group_size)
    quantize_(model, QATConfig(base, step="prepare"), filter_fn=quantizes)
    if cuda:
        model.gradient_checkpointing_enable()
    return _with_lora(model, spec) if lora else model


def _run_steps(  # pragma: no cover
    model: Any, rows: TokenRows, spec: QATSpec, *, device: str
) -> tuple[int, float]:
    """AdamW over ``rows`` in order, warmup then cosine; returns (steps, last step's loss)."""
    import torch  # noqa: PLC0415

    cuda = device != "cpu"
    params = [p for p in model.parameters() if p.requires_grad]
    per_step = spec.micro_batch_size * spec.grad_accum
    total_steps = max(len(rows) // per_step, 1)
    warmup_steps = max(round(total_steps * spec.warmup_fraction), 1)
    optimizer = torch.optim.AdamW(params, lr=spec.learning_rate, betas=ADAM_BETAS, weight_decay=0.0)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: lr_multiplier(step, total_steps=total_steps, warmup_steps=warmup_steps),
    )
    model.train()
    start, loss_value = time.time(), math.nan
    for step in range(total_steps):
        loss_value = 0.0
        for micro in range(spec.grad_accum):
            first = step * per_step + micro * spec.micro_batch_size
            ids = torch.from_numpy(rows[first : first + spec.micro_batch_size]).to(device)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=cuda):
                loss = model(input_ids=ids, labels=ids).loss / spec.grad_accum
            loss.backward()
            loss_value += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP_NORM)
        optimizer.step()
        schedule.step()
        optimizer.zero_grad(set_to_none=True)
        if step % LOG_EVERY_STEPS == 0 or step == total_steps - 1:
            rate = (step + 1) * per_step * spec.seq_len / (time.time() - start)
            print(f"step {step + 1}/{total_steps} loss {loss_value:.4f} {rate:,.0f} tok/s")
    return total_steps, loss_value


def _with_lora(model: Any, spec: QATSpec) -> Any:  # pragma: no cover
    import peft  # noqa: PLC0415

    config = peft.LoraConfig(
        r=spec.lora_rank,
        lora_alpha=spec.lora_alpha,
        lora_dropout=0.0,
        target_modules=list(LORA_TARGETS),
    )
    wrapped = peft.get_peft_model(model, config)
    # Checkpointed blocks need a grad-carrying input when every base weight is frozen.
    wrapped.enable_input_require_grads()
    return wrapped


def _drop_fake_quant(model: Any) -> None:  # pragma: no cover
    """Swap each ``FakeQuantizedLinear`` back to ``nn.Linear`` so the save is a plain HF model."""
    from torchao.quantization.qat.linear import FakeQuantizedLinear  # noqa: PLC0415

    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, FakeQuantizedLinear):
                setattr(parent, name, child.to_linear())
