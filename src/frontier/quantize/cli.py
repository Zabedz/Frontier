"""The ``frontier-quantize`` command: write a variant's compressed-tensors (vLLM), GGUF
(llama.cpp), or torchao QAT checkpoint at ``checkpoint_path``, idempotently, for ``frontier
run`` to serve. One checkpoint per seed of the eval profile, so both commands read their
seeds from one place. A distill variant trains its student from the teacher cache instead,
and ``cache`` builds that cache alone.

The compressed-tensors and GGUF producers are pod-only. QAT and distillation also train on
CPU, which ``--mode smoke`` uses with a capped token budget.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console

from frontier.backends.hf import resolve_device
from frontier.distill.build import deterministic_cublas, ensure_cache
from frontier.distill.cache import TeacherCache
from frontier.distill.fidelity import read_fidelity
from frontier.distill.heldout import BASE_HELDOUT_FILE, read_heldout
from frontier.distill.train import produce_distill
from frontier.pipeline.cli import parse_mode
from frontier.pipeline.config import resolve_config
from frontier.quantize.compressed_tensors import produce_compressed_tensors
from frontier.quantize.gguf import produce_gguf
from frontier.quantize.methods import is_qat_method
from frontier.quantize.qat import produce_qat
from frontier.schema import RunMode, VariantConfig

app = typer.Typer(add_completion=False, help="Produce a variant's served checkpoint.")
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
    max_tokens: Annotated[
        int | None,
        typer.Option(
            "--max-tokens", min=1, help="Cap a QAT or distillation run's training tokens (timing)."
        ),
    ] = None,
    config_root: Annotated[Path, typer.Option("--config-root", help="Config root.")] = Path(
        "configs"
    ),
) -> None:
    """Resolve the config and produce one checkpoint per seed for the config's backend.

    Seed ``s`` calibrates on the draw ``quant.calibration_seed + s``, so every checkpoint is
    reproducible from the config; for a distill variant a seed trains one student. Pass the
    same ``--eval`` as the ``frontier run`` that serves them.
    """
    run_mode = parse_mode(mode)
    resolved = resolve_config(
        config, eval_profile=eval_profile, mode=run_mode, config_root=config_root
    )
    for seed in resolved.eval_spec.seeds:
        out = _produce(
            resolved.variant,
            resolved.backend,
            checkpoints,
            seed=seed,
            mode=run_mode,
            max_tokens=max_tokens,
        )
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
    max_tokens: int | None = None,
) -> Path:
    if variant.family == "distill":
        deterministic_cublas()
        return produce_distill(
            variant,
            checkpoints_root=checkpoints,
            seed=seed,
            device=resolve_device(mode),
            max_tokens=_token_cap(mode, max_tokens),
        )
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
            max_tokens=_token_cap(mode, max_tokens),
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


@app.command()
def cache(
    config: Annotated[
        Path, typer.Option("--config", exists=True, dir_okay=False, help="Variant config YAML.")
    ],
    checkpoints: Annotated[
        Path, typer.Option("--checkpoints", help="Checkpoint root (pod volume).")
    ] = Path("checkpoints"),
    mode: Annotated[str, typer.Option("--mode", help="smoke | full.")] = "full",
    max_tokens: Annotated[
        int | None,
        typer.Option("--max-tokens", min=1, help="Cap the cached training tokens (timing)."),
    ] = None,
    config_root: Annotated[Path, typer.Option("--config-root", help="Config root.")] = Path(
        "configs"
    ),
) -> None:
    """Build or resume a distill variant's teacher cache and check its fidelity; trains nothing."""
    run_mode = parse_mode(mode)
    resolved = resolve_config(config, mode=run_mode, config_root=config_root)
    if resolved.variant.distill is None:
        raise typer.BadParameter(
            f"{resolved.variant.name!r} has no distill block, so it has no teacher cache"
        )
    deterministic_cublas()
    built = ensure_cache(
        resolved.variant,
        checkpoints_root=checkpoints,
        device=resolve_device(run_mode),
        max_tokens=_token_cap(run_mode, max_tokens),
    )
    _summarise_cache(built)


def _summarise_cache(built: TeacherCache) -> None:
    meta = built.meta
    _console.print(f"[green]teacher cache ready[/green]: {built.path}")
    _console.print(
        f"  {meta.n_rows} x {meta.seq_len} tokens ({len(built.training)} training, "
        f"{len(built.heldout)} held out), k={meta.top_k}, teacher {meta.teacher_id} in "
        f"{meta.teacher_dtype} on {meta.device}, deterministic {meta.deterministic}"
    )
    report = read_fidelity(built.path)
    agreement = ", ".join(
        f"{offset}: {share:.3f}" for offset, share in sorted(report.next_token_agreement.items())
    )
    _console.print(
        f"  fidelity over {len(report.rows)} rows, {report.positions} positions: probability "
        f"error {report.prob_error_ratio:.3g}x, rest error {report.rest_error_ratio:.3g}x "
        f"tolerance; next-token agreement by offset {agreement}"
    )
    if meta.rest_summary is not None:
        rest = ", ".join(f"{name} {value:.3g}" for name, value in meta.rest_summary.items())
        _console.print(f"  rest mass: {rest}")
    base = read_heldout(built.path / BASE_HELDOUT_FILE)
    _console.print(
        f"  untrained student on {base.positions} held-out positions: soft KL "
        f"{base.soft_kl:.4f}, teacher top-1 NLL {base.hard_nll:.4f}, corpus NLL "
        f"{base.corpus_nll:.4f}, top-1 agreement {base.top1_agreement:.4f}"
    )


def _require_env(name: str) -> str:  # pragma: no cover
    value = os.environ.get(name)
    if not value:
        raise typer.BadParameter(f"set {name} to the llama.cpp path for a GGUF producer run")
    return value


def _snapshot(model_id: str, revision: str) -> Path:  # pragma: no cover
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    return Path(snapshot_download(model_id, revision=revision))


def _token_cap(mode: RunMode, max_tokens: int | None) -> int | None:
    """``--max-tokens`` when given, else the smoke cap in smoke mode, else no cap."""
    if max_tokens is not None:
        return max_tokens
    return SMOKE_TRAIN_TOKENS if mode == "smoke" else None
