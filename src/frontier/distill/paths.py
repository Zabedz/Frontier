"""Where a distill variant's teacher cache and trained students live under the checkpoint root.

``frontier-quantize`` writes to exactly these paths and ``build_provider`` reads from them.
"""

from __future__ import annotations

import json
from pathlib import Path

from frontier.distill.spec import require_distill_spec, student_model, teacher_model
from frontier.quantize.paths import model_slug
from frontier.schema import RunMode, VariantConfig

STUDENT_METADATA_FILE = "distill.json"


def cache_dir(variant: VariantConfig, *, root: Path) -> Path:
    """``root/distill_cache/<teacher slug>/<corpus>-<train_tokens>t-L<seq_len>-k<top_k>``.

    The name holds every field that changes the cached rows or targets, so arms that differ
    only in their loss share one cache. The teacher revision is checked against ``cache.json``.
    """
    spec = require_distill_spec(variant)
    name = f"{spec.corpus}-{spec.train_tokens}t-L{spec.seq_len}-k{spec.top_k}"
    return root / "distill_cache" / model_slug(teacher_model(variant).model_id) / name


def student_path(variant: VariantConfig, *, root: Path, seed: int = 0) -> Path:
    """``root/distill/<student slug>/<variant name>``, ``-s<seed>`` appended for seeds 1+."""
    name = f"{variant.name}-s{seed}" if seed else variant.name
    return root / "distill" / model_slug(student_model(variant).model_id) / name


def student_markers() -> tuple[str, str]:
    """The files whose presence marks a finished student; the metadata is written last."""
    return "config.json", STUDENT_METADATA_FILE


def require_student(variant: VariantConfig, *, root: Path, seed: int, mode: RunMode) -> Path:
    """The trained student for ``seed``.

    Raises ``FileNotFoundError`` when a marker is missing, naming the path and the
    ``frontier-quantize run`` that produces it, and ``ValueError`` in full mode for a student
    trained on a capped budget.
    """
    path = student_path(variant, root=root, seed=seed)
    missing = [marker for marker in student_markers() if not (path / marker).is_file()]
    if missing:
        raise FileNotFoundError(
            f"no trained student for {variant.name!r} seed {seed} at {path} (missing "
            f"{', '.join(missing)}); train it with `frontier-quantize run --config "
            f"configs/variants/{variant.name}.yaml --mode {mode} --checkpoints {root}` and the "
            f"--eval profile that lists seed {seed}"
        )
    if mode == "full":
        metadata = json.loads((path / STUDENT_METADATA_FILE).read_text(encoding="utf-8"))
        if metadata["capped"]:
            raise ValueError(
                f"student at {path} trained on a capped budget of {metadata['budget_tokens']} "
                f"tokens (--max-tokens or smoke); a full run needs the configured budget"
            )
    return path
