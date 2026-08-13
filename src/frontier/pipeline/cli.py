"""The ``frontier`` command: run one variant x eval-profile and append a result row.

Options use the ``Annotated`` form so a ``typer.Option`` call never sits in a default
argument, which ruff bugbear ``B008`` flags. ``mode`` is a plain string validated in the
body, which holds across typer releases whatever they do with Enums.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich.console import Console
from rich.table import Table

from frontier.io.store import ResultStore
from frontier.metrics.bootstrap import DEFAULT_RESAMPLES
from frontier.metrics.calibration import DEFAULT_BINS
from frontier.pipeline.runner import run as run_pipeline
from frontier.schema import ResultRow, RunMode

if TYPE_CHECKING:
    from frontier.analysis.frontier_chart import ColorBy
    from frontier.analysis.load import XCost
    from frontier.analysis.significance import PairSignificance, Skipped

# The analysis stack pulls matplotlib and pandas, so `plot` imports it in the body to keep
# `frontier run` startup cheap.

_FIGURE_NAMES = frozenset({"frontier", "reliability", "sweep"})

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
    checkpoints: Annotated[
        Path,
        typer.Option("--checkpoints", help="Root of produced Track-B checkpoints (pod volume)."),
    ] = Path("checkpoints"),
    skip_latency: Annotated[
        bool,
        typer.Option(
            "--skip-latency", help="Skip the latency/memory rig (leave those fields empty)."
        ),
    ] = False,
    skip_predictions: Annotated[
        bool,
        typer.Option(
            "--skip-predictions", help="Skip the per-item predictions sidecar (row only)."
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
        checkpoints_root=checkpoints,
        measure_latency=not skip_latency,
        write_predictions=not skip_predictions,
    )
    _summarise(rows, results)


@app.command()
def plot(
    results: Annotated[Path, typer.Option("--results", help="Result store root.")] = Path(
        "results"
    ),
    plots_dir: Annotated[
        Path, typer.Option("--plots-dir", help="Directory the figures are written to.")
    ] = Path("plots"),
    task: Annotated[
        str | None, typer.Option("--task", help="Restrict to one task_name (default: all).")
    ] = None,
    x: Annotated[
        str, typer.Option("--x", help="Frontier cost axis: latency | memory | cost_inv.")
    ] = "memory",
    color_by: Annotated[
        str, typer.Option("--color-by", help="Colour frontier markers by: family | track.")
    ] = "family",
    figures: Annotated[
        str,
        typer.Option(
            "--figures", help="'all' or a comma list from {frontier, reliability, sweep}."
        ),
    ] = "all",
    allow_cross_track: Annotated[
        bool,
        typer.Option(
            "--allow-cross-track",
            help="Draw a non-cross-track x-axis (latency/cost_inv) across both tracks anyway.",
        ),
    ] = False,
) -> None:
    """Read the result store and write the requested figures to ``plots_dir``."""
    from frontier.analysis import (  # noqa: PLC0415
        collapse_seeds,
        ece_bins_sweep_figure,
        frontier_chart,
        load_all_predictions,
        load_tidy,
        prediction_labels,
        reliability_gallery,
    )

    x_axis = _parse_x(x)
    grouping = _parse_color_by(color_by)
    wanted = _parse_figures(figures)
    store = ResultStore(results)
    tidy = load_tidy(store, task_name=task)
    written: list[Path] = []
    if "frontier" in wanted:
        written.append(
            frontier_chart(
                collapse_seeds(tidy),
                x=x_axis,
                color_by=grouping,
                out_path=plots_dir / "frontier.png",
                allow_cross_track=allow_cross_track,
            )
        )
    if wanted & {"reliability", "sweep"}:
        preds = load_all_predictions(tidy, root=results)
        dropped = [label for label in prediction_labels(tidy) if label not in preds]
        if dropped:
            _console.print(f"[yellow]no sidecar for: {', '.join(dropped)}[/yellow]")
        if "reliability" in wanted and preds:
            written.append(
                reliability_gallery(preds, out_path=plots_dir / "reliability-gallery.png")
            )
        if "sweep" in wanted and preds:
            written.append(ece_bins_sweep_figure(preds, out_path=plots_dir / "ece-bins-sweep.png"))
    _summarise_plots(written, plots_dir)


@app.command()
def significance(
    results: Annotated[Path, typer.Option("--results", help="Result store root.")] = Path(
        "results"
    ),
    task: Annotated[
        str | None, typer.Option("--task", help="Restrict to one task_name (default: all).")
    ] = None,
    references: Annotated[
        Path | None,
        typer.Option(
            "--references",
            exists=True,
            dir_okay=False,
            help="Backend-to-reference map (default: configs/analysis/significance.yaml).",
        ),
    ] = None,
    bins: Annotated[
        int, typer.Option("--bins", help="Bin count for the headline ECE delta.")
    ] = DEFAULT_BINS,
    resamples: Annotated[
        int, typer.Option("--resamples", help="Bootstrap resamples per statistic.")
    ] = DEFAULT_RESAMPLES,
    confidence_level: Annotated[
        float, typer.Option("--confidence-level", help="Interval mass, e.g. 0.95.")
    ] = 0.95,
    seed: Annotated[int, typer.Option("--seed", help="Resampling seed.")] = 0,
    out: Annotated[
        Path | None,
        typer.Option("--out", help="Parquet output (default: <results>/significance.parquet)."),
    ] = None,
) -> None:
    """Pair each variant against its backend's reference and bootstrap the deltas.

    The damage gap says whether calibration degrades faster than accuracy; the damage
    ratio says by what multiple.
    """
    from frontier.analysis import (  # noqa: PLC0415
        load_references,
        load_tidy,
        significance_table,
        to_frame,
    )

    store = ResultStore(results)
    tidy = load_tidy(store, task_name=task)
    reference_map = load_references() if references is None else load_references(references)
    found, skipped = significance_table(
        tidy,
        root=results,
        references=reference_map,
        n_bins=bins,
        confidence_level=confidence_level,
        n_resamples=resamples,
        rng=seed,
    )
    destination = out if out is not None else results / "significance.parquet"
    if found:
        # Staged like the result store's parquet, so the previous table survives a crash
        # part way through the write.
        destination.parent.mkdir(parents=True, exist_ok=True)
        staging = destination.with_suffix(destination.suffix + ".tmp")
        to_frame(found).to_parquet(staging)
        staging.replace(destination)
    _summarise_significance(found, skipped, destination if found else None)


def _parse_mode(value: str) -> RunMode:
    if value == "smoke":
        return "smoke"
    if value == "full":
        return "full"
    raise typer.BadParameter("mode must be 'smoke' or 'full'")


def _parse_x(value: str) -> XCost:
    if value == "latency":
        return "latency"
    if value == "memory":
        return "memory"
    if value == "cost_inv":
        return "cost_inv"
    raise typer.BadParameter("x must be one of: latency, memory, cost_inv")


def _parse_color_by(value: str) -> ColorBy:
    if value == "family":
        return "family"
    if value == "track":
        return "track"
    raise typer.BadParameter("color_by must be 'family' or 'track'")


def _parse_figures(value: str) -> frozenset[str]:
    if value.strip() == "all":
        return _FIGURE_NAMES
    parts = frozenset(part.strip() for part in value.split(",") if part.strip())
    unknown = parts - _FIGURE_NAMES
    if not parts or unknown:
        raise typer.BadParameter(
            f"figures must be 'all' or a comma list from {sorted(_FIGURE_NAMES)}"
        )
    return parts


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


def _interval(point: float, low: float, high: float, *, places: int = 4) -> str:
    return f"{point:+.{places}f} [{low:+.{places}f}, {high:+.{places}f}]"


def _summarise_significance(
    found: list[PairSignificance], skipped: list[Skipped], destination: Path | None
) -> None:
    for skip in skipped:
        _console.print(f"[yellow]skipped {skip.variant} on {skip.task}: {skip.reason}[/yellow]")
    if not found:
        _console.print("[yellow]no pairs to test[/yellow]")
        return
    table = Table(title="Paired significance (variant against its backend reference)")
    for column in ("pair", "n", "d accuracy", "d ECE", "damage gap", "damage ratio"):
        table.add_column(column, overflow="fold")
    for item in found:
        ratio = item.damage_ratio
        ratio_cell = (
            _interval(ratio.point, ratio.low, ratio.high, places=2)
            if ratio.usable
            else f"{ratio.point:+.2f} (not estimable)"
        )
        table.add_row(
            item.pair.label,
            str(item.n_items),
            _interval(item.delta_accuracy.point, item.delta_accuracy.low, item.delta_accuracy.high),
            _interval(item.delta_ece.point, item.delta_ece.low, item.delta_ece.high),
            _interval(item.damage_gap.point, item.damage_gap.low, item.damage_gap.high, places=3),
            ratio_cell,
        )
    _console.print(table)
    for item in found:
        verdict = (
            "calibration degrades faster than accuracy"
            if item.damage_gap.excludes_zero and item.damage_gap.point > 0.0
            else "no separation between the two damages"
        )
        _console.print(f"  {item.pair.label}: {verdict}")
        if not item.delta_ece_sign_stable:
            _console.print(
                "    [yellow]the ECE delta changes sign across the bin sweep, so no single "
                "bin count supports a claim[/yellow]"
            )
        elif not item.delta_ece_sweep_all_exclude_zero:
            _console.print(
                "    [yellow]the ECE delta holds its sign across the sweep but its interval "
                "touches zero at one or more bin counts[/yellow]"
            )
        if not item.damage_ratio.usable:
            denominator = item.damage_ratio.denominator
            _console.print(
                f"    ratio not estimable: accuracy damage "
                f"{_interval(denominator.point, denominator.low, denominator.high, places=4)}, "
                f"{item.damage_ratio.nonfinite_resamples} undefined resamples"
            )
    if destination is not None:
        _console.print(f"wrote {destination}")


def _summarise_plots(written: list[Path], plots_dir: Path) -> None:
    if not written:
        _console.print("[yellow]no figures written (empty store or no sidecars)[/yellow]")
        return
    _console.print(f"[green]wrote {len(written)} figure(s)[/green] to [bold]{plots_dir}[/bold]")
    for path in written:
        _console.print(f"  {path}")
