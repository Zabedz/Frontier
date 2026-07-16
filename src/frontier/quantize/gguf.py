"""The GGUF producer: HF snapshot -> f16 GGUF -> k-quant, via llama.cpp tooling.

Two subprocess steps, both CPU (no GPU is needed to build the file): ``convert_hf_to_gguf.py``
writes an f16 intermediate once, then ``llama-quantize`` derives each k-quant from it.
The subprocess runner is injected the way ``io.machine`` injects its nvidia-smi runner, so
the command construction is exercised without llama.cpp present; the real conversion runs
only in the gated pod producer test. Every step is idempotent per output file, so a
re-run of the batch driver does not rebuild a checkpoint that already exists.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from frontier.quantize.paths import checkpoint_path, gguf_quant_type, model_slug
from frontier.schema import VariantConfig

GgufRunner = Callable[[Sequence[str]], None]


def _run(command: Sequence[str]) -> None:  # pragma: no cover
    subprocess.run(list(command), check=True)


def convert_to_gguf(
    model_snapshot: Path,
    *,
    out_path: Path,
    llama_cpp_repo: Path,
    run: GgufRunner | None = None,
) -> Path:
    """Run ``convert_hf_to_gguf.py`` to an f16 GGUF at ``out_path``. Idempotent per file."""
    if out_path.exists():
        return out_path
    execute = run or _run
    out_path.parent.mkdir(parents=True, exist_ok=True)
    execute(
        [
            "python",
            str(llama_cpp_repo / "convert_hf_to_gguf.py"),
            str(model_snapshot),
            "--outfile",
            str(out_path),
            "--outtype",
            "f16",
        ]
    )
    return out_path


def quantize_gguf(
    f16_path: Path,
    *,
    out_path: Path,
    quant_type: str,
    llama_quantize_bin: Path,
    run: GgufRunner | None = None,
) -> Path:
    """Run ``llama-quantize`` from an f16 GGUF to ``quant_type`` at ``out_path``."""
    if out_path.exists():
        return out_path
    execute = run or _run
    out_path.parent.mkdir(parents=True, exist_ok=True)
    execute([str(llama_quantize_bin), str(f16_path), str(out_path), quant_type])
    return out_path


def produce_gguf(
    variant: VariantConfig,
    backend: Mapping[str, Any],
    *,
    checkpoints_root: Path,
    llama_cpp_repo: Path,
    llama_quantize_bin: Path,
    model_snapshot: Path,
    run: GgufRunner | None = None,
) -> Path:
    """Convert once to f16, then quantise to the variant's k-quant. Idempotent per file.

    ``model_snapshot`` is the resolved HF cache path for the base model. The f16
    intermediate is shared across every k-quant of the same model, so it is converted
    once and reused. Returns the final ``.gguf`` path.
    """
    out = checkpoint_path(variant, backend, root=checkpoints_root)
    if out.exists():
        return out
    f16_path = out.parent / f"{model_slug(variant.model.model_id)}.f16.gguf"
    convert_to_gguf(model_snapshot, out_path=f16_path, llama_cpp_repo=llama_cpp_repo, run=run)
    return quantize_gguf(
        f16_path,
        out_path=out,
        quant_type=gguf_quant_type(backend["weight_dtype"]),
        llama_quantize_bin=llama_quantize_bin,
        run=run,
    )
