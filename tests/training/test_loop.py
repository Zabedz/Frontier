"""The shared step plan and the AdamW loop, driven on CPU."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from frontier.pipeline.config import resolve_config
from frontier.training.loop import StepPlan, run_steps, step_plan

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"
QAT_ROWS = 9_765  # 10M tokens / 1024
DISTILL_ROWS = 29_296  # 30M tokens / 1024
TRUE_WEIGHT = (1.5, -2.0, 0.5)
N_ROWS = 64
ROWS_PER_STEP = 16


@dataclass(frozen=True, slots=True)
class _Recipe:
    seq_len: int = 1024
    micro_batch_size: int = 4
    grad_accum: int = 4
    learning_rate: float = 1.0e-5
    warmup_fraction: float = 0.03


def test_the_distill_budget_fills_whole_steps() -> None:
    assert step_plan(DISTILL_ROWS, _Recipe()) == StepPlan(
        total_steps=1_831, warmup_steps=55, rows_per_step=16
    )


def test_the_qat_budget_keeps_its_logged_step_count() -> None:
    assert step_plan(QAT_ROWS, _Recipe()) == StepPlan(
        total_steps=610, warmup_steps=18, rows_per_step=16
    )


def test_fewer_rows_than_one_step_are_refused() -> None:
    with pytest.raises(ValueError, match="cannot fill one 16-row"):
        step_plan(15, _Recipe())


@pytest.mark.parametrize("name", ["student-qat-3bit-full", "distill-softlabel-topk"])
def test_the_shipped_specs_are_step_recipes(name: str) -> None:
    variant = resolve_config(
        CONFIG_ROOT / "variants" / f"{name}.yaml", config_root=CONFIG_ROOT
    ).variant
    spec = variant.qat if variant.qat is not None else variant.distill
    assert spec is not None
    assert step_plan(QAT_ROWS, spec).rows_per_step == ROWS_PER_STEP


def _regression() -> tuple[Any, Any, Any]:
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    features = torch.randn(N_ROWS, len(TRUE_WEIGHT))
    return torch, features, features @ torch.tensor(TRUE_WEIGHT)


def test_run_steps_trains_and_ends_the_schedule_at_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    torch, features, target = _regression()
    recipe = _Recipe(micro_batch_size=2, grad_accum=2, learning_rate=0.05, warmup_fraction=0.1)
    weight = torch.zeros(len(TRUE_WEIGHT), requires_grad=True)
    optimisers: list[Any] = []
    adamw = torch.optim.AdamW

    def recorded(*args: Any, **kwargs: Any) -> Any:
        optimisers.append(adamw(*args, **kwargs))
        return optimisers[-1]

    monkeypatch.setattr(torch.optim, "AdamW", recorded)

    def mse(first: int) -> Any:
        rows = slice(first, first + recipe.micro_batch_size)
        return ((features[rows] @ weight - target[rows]) ** 2).mean()

    with torch.no_grad():
        start_loss = float(((features @ weight - target) ** 2).mean())
    steps, final_loss = run_steps([weight], recipe, n_rows=N_ROWS, micro_loss=mse, device="cpu")
    assert steps == N_ROWS // 4
    assert final_loss < start_loss
    (optimiser,) = optimisers
    assert optimiser.param_groups[0]["lr"] == pytest.approx(0.0, abs=1e-12)


def test_a_non_finite_loss_stops_before_any_update() -> None:
    torch, _features, _target = _regression()
    weight = torch.ones(len(TRUE_WEIGHT), requires_grad=True)

    def poisoned(_first: int) -> Any:
        return (weight * float("nan")).sum()

    with pytest.raises(FloatingPointError, match="step 1/"):
        run_steps(
            [weight],
            _Recipe(micro_batch_size=2, grad_accum=2),
            n_rows=8,
            micro_loss=poisoned,
            device="cpu",
        )
    assert torch.equal(weight.detach(), torch.ones(len(TRUE_WEIGHT)))
