"""QAT producer: torchao fake-quant training, full-parameter or through LoRA.

Full QAT saves a plain HF checkpoint, which ``TorchaoLogitProvider`` quantises on load as it
does the PTQ control. QAT+LoRA saves only its adapter, which the provider attaches beside the
quantised base. The corpus, its packing, and the AdamW loop live in ``frontier.training``,
shared with distillation, and are re-exported here. ``torch``, ``transformers``, ``peft``,
``safetensors``, and ``torchao`` are imported on first use.
"""

from __future__ import annotations

import json
import time
from collections.abc import Iterable, Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from frontier.backends.torchao import (
    ADAPTER_FILE,
    DOWN,
    UP,
    intx_weight_only_config,
    quantizes,
)
from frontier.quantize.methods import QAT_LORA_METHOD, is_qat_method
from frontier.quantize.paths import checkpoint_path
from frontier.schema import QATSpec, VariantConfig
from frontier.training.corpus import (
    TRAINING_CORPORA,
    TextStream,
    TokenRows,
    pack_tokens,
    stream_corpus,
)
from frontier.training.loop import lr_multiplier, run_steps

__all__ = [
    "METADATA_FILE",
    "TRAINING_CORPORA",
    "completion_markers",
    "factors_from_merged",
    "lr_multiplier",
    "pack_tokens",
    "produce_qat",
    "require_qat_spec",
    "save_adapter",
    "stream_corpus",
]

# The projections Qwen2 and Llama-family blocks carry; LoRA wraps every one.
LORA_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
METADATA_FILE = "qat.json"
# peft prefixes the wrapped model's module names with this.
PEFT_PREFIX = "base_model.model."


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
    if out.is_dir() and all((out / marker).exists() for marker in completion_markers(variant)):
        return out
    budget = spec.train_tokens if max_tokens is None else min(max_tokens, spec.train_tokens)
    stream = (texts or stream_corpus)(spec.corpus)
    return _train(variant, spec, out, seed=seed, device=device, budget=budget, texts=stream)


def completion_markers(variant: VariantConfig) -> tuple[str, str]:
    """The files whose presence marks a finished checkpoint for ``variant``."""
    assert variant.quant is not None
    weights = ADAPTER_FILE if variant.quant.method == QAT_LORA_METHOD else "config.json"
    return weights, METADATA_FILE


def factors_from_merged(
    base: Mapping[str, Any], merged: Mapping[str, Any], *, rank: int
) -> dict[str, Any]:
    """Adapter factors recovered from merged LoRA weights, as ``ADAPTER_FILE`` keys.

    ``merged - base`` is the adapter plus the bf16 rounding of the merge; truncating its SVD at
    ``rank`` keeps the adapter and drops most of that rounding. ``base`` and ``merged`` map
    linear names to weights. Raises ``ValueError`` when the two name different layers.
    """
    import torch  # noqa: PLC0415

    if base.keys() != merged.keys():
        differing = sorted(base.keys() ^ merged.keys())
        raise ValueError(f"base and merged weights name different layers: {differing}")
    factors: dict[str, Any] = {}
    for name in base:
        delta = merged[name].to(torch.float32) - base[name].to(torch.float32)
        u, s, vh = torch.linalg.svd(delta, full_matrices=False)
        factors[f"{name}.{UP}"] = (u[:, :rank] * s[:rank]).cpu().contiguous()
        factors[f"{name}.{DOWN}"] = vh[:rank].cpu().contiguous()
    return factors


def save_adapter(factors: Mapping[str, Any], out: Path) -> Path:
    """Write ``factors`` to ``out / ADAPTER_FILE`` as fp32 and return the path."""
    import torch  # noqa: PLC0415
    from safetensors.torch import save_file  # noqa: PLC0415

    path = out / ADAPTER_FILE
    save_file({k: v.to(torch.float32).contiguous() for k, v in factors.items()}, str(path))
    return path


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

    out.mkdir(parents=True, exist_ok=True)
    if lora:
        save_adapter(_lora_factors(model), out)
    else:
        _drop_fake_quant(model)
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
    """``run_steps`` over packed rows with HF's own next-token loss."""
    import torch  # noqa: PLC0415

    def micro_loss(first: int) -> Any:
        ids = torch.from_numpy(rows[first : first + spec.micro_batch_size]).to(device)
        return model(input_ids=ids, labels=ids).loss

    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    return run_steps(params, spec, n_rows=len(rows), micro_loss=micro_loss, device=device)


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


def _lora_factors(model: Any) -> dict[str, Any]:  # pragma: no cover
    """The trained adapter as ``ADAPTER_FILE`` keys: up = B x scaling, down = A."""
    factors: dict[str, Any] = {}
    for name, module in model.named_modules():
        if not hasattr(module, "lora_A") or "default" not in module.lora_A:
            continue
        layer = name.removeprefix(PEFT_PREFIX)
        scaling = module.scaling["default"]
        factors[f"{layer}.{UP}"] = module.lora_B["default"].weight.detach().float().cpu() * scaling
        factors[f"{layer}.{DOWN}"] = module.lora_A["default"].weight.detach().float().cpu()
    return factors


def _drop_fake_quant(model: Any) -> None:  # pragma: no cover
    """Swap each ``FakeQuantizedLinear`` back to ``nn.Linear`` so the save is a plain HF model."""
    from torchao.quantization.qat.linear import FakeQuantizedLinear  # noqa: PLC0415

    for parent in list(model.modules()):
        for name, child in list(parent.named_children()):
            if isinstance(child, FakeQuantizedLinear):
                setattr(parent, name, child.to_linear())
