"""The `frontier plot` command reads a seeded store and writes the figures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from typer.testing import CliRunner

from frontier.io.predictions import PredictionRows, predictions_key, write_predictions_rows
from frontier.io.store import ResultStore, append_row
from frontier.pipeline.cli import app
from frontier.schema import (
    Backend,
    Latency,
    MachineState,
    Memory,
    Provenance,
    Quality,
    ResultRow,
    TaskSpec,
)

_MACHINE = MachineState(
    gpu_clock_sm_mhz=1500,
    gpu_clock_mem_mhz=6000,
    gpu_temp_c=61,
    power_w=118.5,
    clocks_locked=False,
    clock_drift_flag=False,
)


def _row(
    *, name: str, family: str, track: str, config_hash: str, accuracy: float, vram: float
) -> ResultRow:
    provenance = Provenance(
        git_sha="sha",
        config_hash=config_hash,
        model_id="Qwen/Qwen2.5-3B-Instruct",
        model_revision="main",
        hardware_id="gpu:test",
        driver_version="550",
        cuda_version="12.4",
        seed=0,
        timestamp="2026-07-13T00:00:00+00:00",
    )
    backend = Backend(
        inference_backend="hf",
        backend_version="5.0.0",
        weight_dtype="int4",
        kv_cache_dtype="fp16",
        gpu_offload_layers=-1,
        track=track,  # type: ignore[arg-type]
    )
    task = TaskSpec("mmlu", "test", 500, "zeroshot", "letter_softmax", "cyclic", "redux", False)
    quality = Quality(
        accuracy=accuracy,
        accuracy_ci_low=accuracy - 0.02,
        accuracy_ci_high=accuracy + 0.02,
        ece_equal_width=0.05,
        ece_equal_mass_ace=0.045,
        ece_bin_sweep={5: 0.04, 10: 0.05},
        ece_ci_low=0.03,
        ece_ci_high=0.07,
        brier=0.2,
        brier_reliability=0.02,
        brier_resolution=0.15,
        brier_uncertainty=0.37,
        nll=0.5,
        perplexity=float("nan"),
        temperature_scaled=False,
        temperature=1.0,
    )
    latency = Latency(1, 10.0, 12.0, 3.0, 4.0, 200.0, 20, 5, _MACHINE)
    memory = Memory(1, 512, vram, 3000.0, 3200.0, 400.0)
    return ResultRow(
        provenance=provenance,
        backend=backend,
        variant_name=name,
        family=family,  # type: ignore[arg-type]
        task=task,
        quality=quality,
        latency=[latency],
        memory=[memory],
        tok_s_per_gb=200.0 / (vram / 1000.0),
        robustness=None,
    )


def _preds(seed: int, n: int, low: float) -> PredictionRows:
    rng = np.random.default_rng(seed)
    confidence = rng.uniform(low, 1.0, size=n)
    correct = rng.uniform(0.0, 1.0, size=n) < confidence
    gold = rng.integers(0, 4, size=n).astype(np.intp)
    predicted = np.where(correct, gold, (gold + 1) % 4).astype(np.intp)
    return PredictionRows(confidence, correct.astype(np.bool_), gold, predicted)


def _seed_store(root: Path) -> None:
    store = ResultStore(root)
    specs = [
        ("fp16", "baseline", "A", "a" * 64, 0.82, 8000.0),
        ("int4-gptq", "ptq", "B", "b" * 64, 0.79, 4000.0),
    ]
    for index, (name, family, track, config_hash, accuracy, vram) in enumerate(specs):
        append_row(
            _row(
                name=name,
                family=family,
                track=track,
                config_hash=config_hash,
                accuracy=accuracy,
                vram=vram,
            ),
            store,
        )
        write_predictions_rows(
            _preds(index, 500, 0.4),
            root=root,
            key=predictions_key(config_hash, 0, "mmlu"),
        )


def test_plot_writes_the_frontier_and_gallery(tmp_path: Path) -> None:
    store_root = tmp_path / "results"
    plots_dir = tmp_path / "plots"
    _seed_store(store_root)

    result = CliRunner().invoke(
        app, ["plot", "--results", str(store_root), "--plots-dir", str(plots_dir)]
    )
    assert result.exit_code == 0, result.output
    assert (plots_dir / "frontier.png").stat().st_size > 0
    assert (plots_dir / "reliability-gallery.png").stat().st_size > 0
    assert (plots_dir / "ece-bins-sweep.png").stat().st_size > 0


def test_plot_rejects_a_bad_x_axis(tmp_path: Path) -> None:
    store_root = tmp_path / "results"
    _seed_store(store_root)
    result = CliRunner().invoke(app, ["plot", "--results", str(store_root), "--x", "bogus"])
    assert result.exit_code != 0
