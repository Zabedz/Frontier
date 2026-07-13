"""The ``frontier`` command: run one variant x eval-profile and append a result row.

Options use the ``Annotated`` form so a ``typer.Option`` call never sits in a default
argument (which ruff bugbear ``B008`` flags). ``mode`` is a plain string validated in
the body, version-stable across typer releases without an Enum assumption.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from frontier.pipeline.runner import run as run_pipeline
from frontier.schema import ResultRow, RunMode

app = typer.Typer(
    add_completion=False,
    help="Run one variant x eval-profile and append a result row.",
)
_console = Console()


@app.callback()
def _main() -> None:
    """Frontier: score one compressed variant and append its result row."""


@app.command()
def run(
    config: Annotated[
        Path, typer.Option("--config", exists=True, dir_okay=False, help="Variant config YAML.")
    ],
    eval_profile: Annotated[
        str | None,
        typer.Option("--eval", help="Eval profile under configs/evals/ (default: base's eval)."),
    ] = None,
    mode: Annotated[str, typer.Option("--mode", help="smoke | full.")] = "full",
    results: Annotated[Path, typer.Option("--results", help="Result store root.")] = Path(
        "results"
    ),
    config_root: Annotated[Path, typer.Option("--config-root", help="Config root.")] = Path(
        "configs"
    ),
    skip_latency: Annotated[
        bool,
        typer.Option(
            "--skip-latency", help="Skip the latency/memory rig (leave those fields empty)."
        ),
    ] = False,
) -> None:
    """Resolve, score, and append one row per seed."""
    rows = run_pipeline(
        config,
        eval_profile=eval_profile,
        mode=_parse_mode(mode),
        config_root=config_root,
        results_root=results,
        measure_latency=not skip_latency,
    )
    _summarise(rows, results)


def _parse_mode(value: str) -> RunMode:
    if value == "smoke":
        return "smoke"
    if value == "full":
        return "full"
    raise typer.BadParameter("mode must be 'smoke' or 'full'")


def _summarise(rows: list[ResultRow], results: Path) -> None:
    if not rows:
        _console.print("[yellow]no rows produced[/yellow]")
        return
    first = rows[0]
    _console.print(
        f"[green]appended {len(rows)} row(s)[/green] for [bold]{first.variant_name}[/bold] "
        f"on {first.task.task_name} (n={first.task.num_items})"
    )
    for row in rows:
        _console.print(
            f"  seed {row.provenance.seed}: "
            f"accuracy={row.quality.accuracy:.4f} "
            f"ece_equal_width={row.quality.ece_equal_width:.4f}"
        )
    _console.print(f"store: {results / 'results.parquet'}")
