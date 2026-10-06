"""The ``frontier-quantize`` command: write a variant's compressed-tensors (vLLM) or GGUF
(llama.cpp) checkpoint at ``checkpoint_path``, idempotently, for ``frontier run`` to serve.
One checkpoint per seed of the eval profile, so both commands read their seeds from one place.

A pod command: the compressed-tensors producer runs a GPU calibration pass and the GGUF
producer shells out to llama.cpp, so the body is exercised on the pod.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console

from frontier.pipeline.config import resolve_config
from frontier.quantize.compressed_tensors import produce_compressed_tensors
from frontier.quantize.gguf import produce_gguf
from frontier.schema import VariantConfig

app = typer.Typer(add_completion=False, help="Produce a variant's Track-B checkpoint.")
_console = Console()

_LLAMA_REPO_ENV = "FRONTIER_LLAMA_CPP_REPO"
_LLAMA_QUANTIZE_ENV = "FRONTIER_LLAMA_QUANTIZE_BIN"


@app.callback()
def _main() -> None:
    """Frontier: quantise one variant into its served checkpoint."""


@app.command()
def run(
    config: Annotated[
        Path, typer.Option("--config", exists=True, dir_okay=False, help="Variant config YAML.")
    ],
    checkpoints: Annotated[
        Path, typer.Option("--checkpoints", help="Checkpoint root (pod volume).")
    ] = Path("checkpoints"),
    eval_profile: Annotated[
        str | None,
        typer.Option("--eval", help="Eval profile whose seeds to produce (default: base's)."),
    ] = None,
    config_root: Annotated[Path, typer.Option("--config-root", help="Config root.")] = Path(
        "configs"
    ),
) -> None:
    """Resolve the config and produce one checkpoint per seed for the config's backend.

    Seed ``s`` calibrates on the draw ``quant.calibration_seed + s``, so every checkpoint is
    reproducible from the config. Pass the same ``--eval`` as the ``frontier run`` that
    serves them.
    """
    resolved = resolve_config(config, eval_profile=eval_profile, config_root=config_root)
    for seed in resolved.eval_spec.seeds:
        out = _produce(resolved.variant, resolved.backend, checkpoints, seed=seed)
        _console.print(
            f"[green]checkpoint ready[/green] for [bold]{resolved.variant.name}[/bold] "
            f"seed {seed}: {out}"
        )


def _produce(
    variant: VariantConfig, backend: Mapping[str, Any], checkpoints: Path, *, seed: int
) -> Path:
    inference_backend = backend["inference_backend"]
    if inference_backend == "vllm":
        return produce_compressed_tensors(variant, backend, checkpoints_root=checkpoints, seed=seed)
    if inference_backend == "llama_cpp":
        return produce_gguf(
            variant,
            backend,
            checkpoints_root=checkpoints,
            llama_cpp_repo=Path(_require_env(_LLAMA_REPO_ENV)),
            llama_quantize_bin=Path(_require_env(_LLAMA_QUANTIZE_ENV)),
            model_snapshot=_snapshot(variant.model.model_id, variant.model.model_revision),
        )
    raise typer.BadParameter(
        f"backend {inference_backend!r} has no producer; only vllm and llama_cpp are quantised"
    )


def _require_env(name: str) -> str:  # pragma: no cover
    value = os.environ.get(name)
    if not value:
        raise typer.BadParameter(f"set {name} to the llama.cpp path for a GGUF producer run")
    return value


def _snapshot(model_id: str, revision: str) -> Path:  # pragma: no cover
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    return Path(snapshot_download(model_id, revision=revision))
