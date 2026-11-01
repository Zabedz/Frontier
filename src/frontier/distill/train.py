"""The distillation trainer: one full fine-tune of the student per seed on the cached rows,
fp32 in training and saved in fp16; the modes differ only in the per-position target."""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from frontier.distill.build import cache_request, ensure_cache, served_dtype
from frontier.distill.cache import (
    CacheRequest,
    TeacherCache,
    device_name,
    library_versions,
    tokenizer_fingerprint,
    write_json,
)
from frontier.distill.heldout import HELDOUT_FILE, heldout_report, write_heldout
from frontier.distill.losses import (
    LOSS_CHUNK_POSITIONS,
    chunked_mean,
    per_position_loss,
    target_tensors,
)
from frontier.distill.paths import STUDENT_METADATA_FILE, student_markers, student_path
from frontier.distill.spec import require_distill_spec, student_model
from frontier.schema import DistillSpec, VariantConfig
from frontier.training.corpus import TextStream
from frontier.training.loop import run_steps, step_plan

SAVED_DTYPE = "float16"


def produce_distill(
    variant: VariantConfig,
    *,
    checkpoints_root: Path,
    seed: int,
    device: str,
    max_tokens: int | None = None,
    texts: TextStream | None = None,
) -> Path:
    """Train ``variant``'s student for ``seed`` and save it at ``student_path``; idempotent.

    Builds the teacher cache first when it is missing. ``max_tokens`` caps the budget (smoke,
    timing). ``texts`` replaces the streamed corpus, for tests. Raises ``ValueError`` when a
    finished checkpoint at the path was trained with another recipe or budget.
    """
    spec = require_distill_spec(variant)
    out = student_path(variant, root=checkpoints_root, seed=seed)
    request = cache_request(variant, max_tokens=max_tokens)
    if all((out / marker).is_file() for marker in student_markers()):
        _check_recipe(out, variant, spec, request.budget_tokens)
        return out
    # A budget under one optimiser step fails here, before the teacher runs.
    step_plan(request.train_rows, spec)
    cache = ensure_cache(
        variant,
        checkpoints_root=checkpoints_root,
        device=device,
        max_tokens=max_tokens,
        texts=texts,
    )
    return _train(variant, spec, cache, out, request=request, seed=seed, device=device)


def _recipe(variant: VariantConfig, spec: DistillSpec) -> dict[str, dict[str, Any]]:
    return {"model": asdict(variant.model), "distill": asdict(spec)}


def _check_recipe(out: Path, variant: VariantConfig, spec: DistillSpec, budget: int) -> None:
    stored = json.loads((out / STUDENT_METADATA_FILE).read_text(encoding="utf-8"))
    current = _recipe(variant, spec)
    for section, fields in current.items():
        for name, value in fields.items():
            recorded = stored["recipe"][section].get(name)
            if recorded != value:
                raise ValueError(
                    f"student at {out} was trained with {section}.{name}={recorded!r}; this "
                    f"run has {value!r}, so delete it or pass another --checkpoints"
                )
    if stored["budget_tokens"] != budget:
        raise ValueError(
            f"student at {out} was trained on {stored['budget_tokens']} tokens; this run's "
            f"budget is {budget}, so delete it or pass another --checkpoints"
        )


def _train(
    variant: VariantConfig,
    spec: DistillSpec,
    cache: TeacherCache,
    out: Path,
    *,
    request: CacheRequest,
    seed: int,
    device: str,
) -> Path:
    import torch  # noqa: PLC0415
    import transformers  # noqa: PLC0415

    student = student_model(variant)
    torch.manual_seed(seed)
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        student.model_id, revision=student.revision
    )
    if tokenizer_fingerprint(tokenizer) != cache.meta.tokenizer_fingerprint:
        raise ValueError(
            f"student tokenizer {student.model_id} differs from the one cache {cache.path} "
            f"was built with"
        )
    model = transformers.AutoModelForCausalLM.from_pretrained(
        student.model_id, revision=student.revision, dtype=torch.float32
    ).to(device)
    if int(model.config.vocab_size) != cache.meta.vocab_size:
        raise ValueError(
            f"student {student.model_id} has {model.config.vocab_size} vocabulary rows; the "
            f"cache's teacher has {cache.meta.vocab_size}"
        )
    model.config.use_cache = False
    if device != "cpu":
        model.gradient_checkpointing_enable()
    n_rows = request.train_rows
    if n_rows > len(cache.training):
        raise ValueError(
            f"{n_rows} training rows requested; cache {cache.path} holds {len(cache.training)} "
            f"before its held-out rows"
        )
    # The seed reorders the rows and seeds torch, as for QAT; the rows themselves are fixed.
    order = np.random.default_rng(seed).permutation(n_rows)
    decoder, head = model.get_decoder(), model.get_output_embeddings()
    per_position = per_position_loss(spec.mode)

    def micro_loss(first: int) -> Any:
        batch = cache.batch(order[first : first + spec.micro_batch_size], spec.mode)
        ids = torch.from_numpy(batch.input_ids).to(device)
        hidden = decoder(input_ids=ids).last_hidden_state[:, :-1]
        targets = target_tensors(batch, spec.mode, device=device)
        return chunked_mean(hidden, head, per_position, *targets)

    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    start = time.time()
    steps, final_loss = run_steps(params, spec, n_rows=n_rows, micro_loss=micro_loss, device=device)
    wall_seconds = round(time.time() - start, 1)

    model.eval()
    model.to(torch.float16)
    # The saved config serves like the hub's: the KV cache was off for training only.
    model.config.use_cache = True
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    tokenizer.save_pretrained(out)
    # Scored as the hf backend will serve the saved weights: fp16 on CUDA, upcast on CPU.
    model.to(getattr(torch, served_dtype(device)))
    report = heldout_report(model, cache, student_id=student.model_id, device=device)
    write_heldout(out / HELDOUT_FILE, report)
    rows_used = steps * spec.micro_batch_size * spec.grad_accum
    metadata = {
        "variant": variant.name,
        "seed": seed,
        "mode": spec.mode,
        "recipe": _recipe(variant, spec),
        "budget_tokens": request.budget_tokens,
        "capped": request.capped,
        "rows_used": rows_used,
        "steps": steps,
        "tokens_trained": rows_used * spec.seq_len,
        "positions_per_row": spec.seq_len - 1,
        "loss_chunk_positions": LOSS_CHUNK_POSITIONS,
        "final_loss": final_loss,
        "wall_seconds": wall_seconds,
        "cache": {
            "path": str(cache.path),
            "rows_fingerprint": cache.meta.rows_fingerprint,
            "n_rows": cache.meta.n_rows,
        },
        "saved_dtype": SAVED_DTYPE,
        "device": device_name(device),
        "versions": library_versions(),
    }
    write_json(out / STUDENT_METADATA_FILE, metadata)
    return out
