"""What a distill variant names: its teacher, its student, and the recipe the cache can train.

Pure Python over ``schema``, so config resolution, serving, and provenance import it freely.
"""

from __future__ import annotations

from dataclasses import dataclass

from frontier.schema import DistillSpec, VariantConfig

# The cache stores the teacher's probabilities at T=1; the temperature axis is fitted post hoc.
CACHED_KD_TEMPERATURE = 1.0
# The budget pays for one pass over the cached rows.
DISTILL_EPOCHS = 1
# The hf weight_dtype a student is saved, scored and served in, and the one the teacher's fp16
# reference row is served in: float16 on CUDA, float32 on CPU.
SERVED_WEIGHT_DTYPE = "fp16"


@dataclass(frozen=True, slots=True)
class ModelRef:
    """A hub model and the revision it loads at."""

    model_id: str
    revision: str


def validate_distill(variant: VariantConfig) -> None:
    """Refuse a distill block the cache cannot train; called at resolve time.

    Raises ``ValueError`` for family ``distill`` without a ``distill`` block or a
    ``model.student_id``, a ``distill`` block on another family, ``kd_temperature != 1.0``,
    or ``epochs != 1``.
    """
    spec = variant.distill
    if variant.family != "distill":
        if spec is not None:
            raise ValueError(
                f"variant {variant.name!r} is family {variant.family!r} but has a distill block"
            )
        return
    if spec is None:
        raise ValueError(f"distill variant {variant.name!r} has no distill block")
    student_model(variant)
    if spec.kd_temperature != CACHED_KD_TEMPERATURE:
        raise ValueError(
            f"variant {variant.name!r} sets kd_temperature {spec.kd_temperature}, but the cache "
            f"holds the teacher's T={CACHED_KD_TEMPERATURE:g} probabilities; fit a temperature "
            f"to the trained student with `frontier recalibrate` instead"
        )
    if spec.epochs != DISTILL_EPOCHS:
        raise ValueError(
            f"variant {variant.name!r} sets epochs {spec.epochs}; the budget caps distillation "
            f"at one pass over the cached rows"
        )


def require_distill_spec(variant: VariantConfig) -> DistillSpec:
    """The variant's ``distill`` block; raises ``ValueError`` when ``model.student_id`` is unset."""
    if variant.distill is None:
        raise ValueError(f"variant {variant.name!r} has no distill block")
    student_model(variant)
    return variant.distill


def token_budget(spec: DistillSpec, max_tokens: int | None) -> int:
    """The training tokens a run uses: ``train_tokens``, capped by ``max_tokens`` when given."""
    return spec.train_tokens if max_tokens is None else min(max_tokens, spec.train_tokens)


def teacher_model(variant: VariantConfig) -> ModelRef:
    """``model.model_id`` at ``model.model_revision``."""
    return ModelRef(variant.model.model_id, variant.model.model_revision)


def student_model(variant: VariantConfig) -> ModelRef:
    """``model.student_id`` at ``model.student_revision``; raises ``ValueError`` when unset."""
    if variant.model.student_id is None:
        raise ValueError(f"variant {variant.name!r} names no model.student_id to distil into")
    return ModelRef(variant.model.student_id, variant.model.student_revision)


def scored_model(variant: VariantConfig) -> ModelRef:
    """The hub model a row's weights come from, and so its ``provenance.model_id``.

    The student for a distill variant, ``model.model_id`` for every other family.
    """
    if variant.family == "distill":
        return student_model(variant)
    return teacher_model(variant)
