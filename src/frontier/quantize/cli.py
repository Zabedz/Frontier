"""The ``frontier-quantize`` command: write a variant's compressed-tensors (vLLM), GGUF
(llama.cpp), or torchao QAT checkpoint at ``checkpoint_path``, idempotently, for ``frontier
run`` to serve. One checkpoint per seed of the eval profile, so both commands read their
seeds from one place.

The compressed-tensors and GGUF producers are pod-only. QAT also trains on CPU, which
``--mode smoke`` uses with a capped token budget.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console

from frontier.backends.hf import resolve_device
from frontier.pipeline.cli import parse_mode
from frontier.pipeline.config import resolve_config
from frontier.quantize.compressed_tensors import produce_compressed_tensors
from frontier.quantize.gguf import produce_gguf
from frontier.quantize.methods import is_qat_method
from frontier.quantize.qat import produce_qat
from frontier.schema import RunMode, VariantConfig

app = typer.Typer(add_completion=False, help="Produce a variant's Track-B checkpoint.")
_console = Console()

_LLAMA_REPO_ENV = "FRONTIER_LLAMA_CPP_REPO"
_LLAMA_QUANTIZE_ENV = "FRONTIER_LLAMA_QUANTIZE_BIN"
# One optimiser step at the configured 16 x 1024 batch: enough to prove the loop on a laptop.
SMOKE_TRAIN_TOKENS = 16_384


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
    mode: Annotated[str, typer.Option("--mode", help="smoke | full.")] = "full",
    config_root: Annotated[Path, typer.Option("--config-root", help="Config root.")] = Path(
        "configs"
    ),
) -> None:
    """Resolve the config and produce one checkpoint per seed for the config's backend.

    Seed ``s`` calibrates on the draw ``quant.calibration_seed + s``, so every checkpoint is
    reproducible from the config. Pass the same ``--eval`` as the ``frontier run`` that
    serves them.
    """
    run_mode = parse_mode(mode)
    resolved = resolve_config(
        config, eval_profile=eval_profile, mode=run_mode, config_root=config_root
    )
    for seed in resolved.eval_spec.seeds:
        out = _produce(resolved.variant, resolved.backend, checkpoints, seed=seed, mode=run_mode)
        _console.print(
            f"[green]checkpoint ready[/green] for [bold]{resolved.variant.name}[/bold] "
            f"seed {seed}: {out}"
        )


def _produce(
    variant: VariantConfig,
    backend: Mapping[str, Any],
    checkpoints: Path,
    *,
    seed: int,
    mode: RunMode = "full",
) -> Path:
    inference_backend = backend["inference_backend"]
    if inference_backend == "torchao":
        if variant.quant is None or not is_qat_method(variant.quant.method):
            raise typer.BadParameter(
                f"{variant.name!r} quantises on load in `frontier run`; nothing to produce"
            )
        return produce_qat(
            variant,
            backend,
            checkpoints_root=checkpoints,
            seed=seed,
            device=resolve_device(mode),
            max_tokens=SMOKE_TRAIN_TOKENS if mode == "smoke" else None,
        )
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
