"""The GGUF producer seams (converter argv, quantiser argv, per-file idempotence).

A fake subprocess runner drives them, so no llama.cpp needs to be present.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from frontier.pipeline.config import resolve_config
from frontier.quantize.gguf import convert_to_gguf, produce_gguf, quantize_gguf

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"


class _RecordingRunner:
    """Captures every argv it is handed and creates nothing, so idempotence is testable."""

    def __init__(self) -> None:
        self.commands: list[list[str]] = []

    def __call__(self, command: Sequence[str]) -> None:
        self.commands.append(list(command))


def test_convert_builds_convert_hf_to_gguf_argv(tmp_path: Path) -> None:
    runner = _RecordingRunner()
    out = tmp_path / "model.f16.gguf"
    result = convert_to_gguf(
        tmp_path / "snapshot", out_path=out, llama_cpp_repo=Path("/repo"), run=runner
    )
    assert result == out
    assert runner.commands == [
        [
            "python",
            "/repo/convert_hf_to_gguf.py",
            str(tmp_path / "snapshot"),
            "--outfile",
            str(out),
            "--outtype",
            "f16",
        ]
    ]


def test_convert_is_idempotent(tmp_path: Path) -> None:
    runner = _RecordingRunner()
    out = tmp_path / "model.f16.gguf"
    out.write_bytes(b"already here")
    convert_to_gguf(tmp_path / "snapshot", out_path=out, llama_cpp_repo=Path("/repo"), run=runner)
    assert runner.commands == []


def test_quantize_builds_llama_quantize_argv(tmp_path: Path) -> None:
    runner = _RecordingRunner()
    f16 = tmp_path / "model.f16.gguf"
    out = tmp_path / "model.q4_k_m.gguf"
    result = quantize_gguf(
        f16,
        out_path=out,
        quant_type="Q4_K_M",
        llama_quantize_bin=Path("/bin/llama-quantize"),
        run=runner,
    )
    assert result == out
    assert runner.commands == [["/bin/llama-quantize", str(f16), str(out), "Q4_K_M"]]


def test_produce_gguf_converts_then_quantizes(tmp_path: Path) -> None:
    resolved = resolve_config(
        CONFIG_ROOT / "variants" / "gguf-q4_k_m.yaml", config_root=CONFIG_ROOT
    )
    runner = _RecordingRunner()
    out = produce_gguf(
        resolved.variant,
        resolved.backend,
        checkpoints_root=tmp_path,
        llama_cpp_repo=Path("/repo"),
        llama_quantize_bin=Path("/bin/llama-quantize"),
        model_snapshot=tmp_path / "snapshot",
        run=runner,
    )
    assert out == tmp_path / "gguf" / "Qwen2.5-3B-Instruct.q4_k_m.gguf"
    f16 = tmp_path / "gguf" / "Qwen2.5-3B-Instruct.f16.gguf"
    assert [command[0] for command in runner.commands] == ["python", "/bin/llama-quantize"]
    assert runner.commands[0][2] == str(tmp_path / "snapshot")
    assert runner.commands[0][4] == str(f16)
    assert runner.commands[1] == ["/bin/llama-quantize", str(f16), str(out), "Q4_K_M"]


def test_produce_gguf_is_idempotent_when_output_exists(tmp_path: Path) -> None:
    resolved = resolve_config(
        CONFIG_ROOT / "variants" / "gguf-q4_k_m.yaml", config_root=CONFIG_ROOT
    )
    out = tmp_path / "gguf" / "Qwen2.5-3B-Instruct.q4_k_m.gguf"
    out.parent.mkdir(parents=True)
    out.write_bytes(b"already quantised")
    runner = _RecordingRunner()
    result = produce_gguf(
        resolved.variant,
        resolved.backend,
        checkpoints_root=tmp_path,
        llama_cpp_repo=Path("/repo"),
        llama_quantize_bin=Path("/bin/llama-quantize"),
        model_snapshot=tmp_path / "snapshot",
        run=runner,
    )
    assert result == out
    assert runner.commands == []
