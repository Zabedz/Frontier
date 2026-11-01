"""The AdamW loop both trainers run: linear warmup, cosine decay, clipped gradients, bf16
autocast on CUDA. ``torch`` is imported on first use.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

GRAD_CLIP_NORM = 1.0
ADAM_BETAS = (0.9, 0.95)
LOG_EVERY_STEPS = 10

MicroLoss = Callable[[int], Any]  # first row offset of the micro-batch -> scalar loss tensor


class StepRecipe(Protocol):
    """The schedule fields ``QATSpec`` and ``DistillSpec`` share."""

    @property
    def seq_len(self) -> int: ...
    @property
    def micro_batch_size(self) -> int: ...
    @property
    def grad_accum(self) -> int: ...
    @property
    def learning_rate(self) -> float: ...
    @property
    def warmup_fraction(self) -> float: ...


@dataclass(frozen=True, slots=True)
class StepPlan:
    """Whole optimiser steps over a set of rows; rows past the last whole step are not read."""

    total_steps: int
    warmup_steps: int
    rows_per_step: int


def step_plan(n_rows: int, recipe: StepRecipe) -> StepPlan:
    """Whole steps over ``n_rows``; raises ``ValueError`` when one step needs more rows."""
    rows_per_step = recipe.micro_batch_size * recipe.grad_accum
    total_steps = n_rows // rows_per_step
    if total_steps == 0:
        raise ValueError(f"{n_rows} rows cannot fill one {rows_per_step}-row optimiser step")
    warmup_steps = max(round(total_steps * recipe.warmup_fraction), 1)
    return StepPlan(total_steps=total_steps, warmup_steps=warmup_steps, rows_per_step=rows_per_step)


def lr_multiplier(step: int, *, total_steps: int, warmup_steps: int) -> float:
    """Linear warmup to 1, then cosine decay to 0 at ``total_steps``."""
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
    return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def run_steps(
    params: Sequence[Any],
    recipe: StepRecipe,
    *,
    n_rows: int,
    micro_loss: MicroLoss,
    device: str,
) -> tuple[int, float]:
    """AdamW over ``n_rows`` in order, warmup then cosine; returns (steps, last step's loss).

    ``micro_loss(first)`` returns the mean loss over rows ``first`` to ``first +
    micro_batch_size``; the caller puts the model in train mode. Raises
    ``FloatingPointError`` on a non-finite step loss, before the update.
    """
    import torch  # noqa: PLC0415

    plan = step_plan(n_rows, recipe)
    cuda = device != "cpu"
    optimizer = torch.optim.AdamW(
        params, lr=recipe.learning_rate, betas=ADAM_BETAS, weight_decay=0.0
    )
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: lr_multiplier(
            step, total_steps=plan.total_steps, warmup_steps=plan.warmup_steps
        ),
    )
    start, loss_value = time.time(), math.nan
    for step in range(plan.total_steps):
        loss_value = 0.0
        for micro in range(recipe.grad_accum):
            first = step * plan.rows_per_step + micro * recipe.micro_batch_size
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=cuda):
                loss = micro_loss(first) / recipe.grad_accum
            loss.backward()
            loss_value += float(loss.detach())
        if not math.isfinite(loss_value):
            raise FloatingPointError(
                f"step {step + 1}/{plan.total_steps} loss is {loss_value}; stopped before the "
                f"update"
            )
        torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP_NORM)
        optimizer.step()
        schedule.step()
        optimizer.zero_grad(set_to_none=True)
        if step % LOG_EVERY_STEPS == 0 or step == plan.total_steps - 1:
            rate = (step + 1) * plan.rows_per_step * recipe.seq_len / (time.time() - start)
            print(f"step {step + 1}/{plan.total_steps} loss {loss_value:.4f} {rate:,.0f} tok/s")
    return plan.total_steps, loss_value
