"""The QAT producer's CPU-checkable parts, and a live SmolLM2 train-save-serve loop."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator, Mapping
from dataclasses import replace
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from frontier.backends.torchao import ADAPTER_FILE
from frontier.pipeline.config import resolve_config
from frontier.quantize.paths import checkpoint_path
from frontier.quantize.qat import (
    METADATA_FILE,
    completion_markers,
    factors_from_merged,
    lr_multiplier,
    pack_tokens,
    produce_qat,
    require_qat_spec,
)

CONFIG_ROOT = Path(__file__).resolve().parents[2] / "configs"
EOS = 0
SEQ_LEN = 4
TOTAL_STEPS = 100
WARMUP_STEPS = 10
# Sentences for the live loop: enough tokens for one 16 x 1024 step on SmolLM2's tokenizer.
LIVE_TEXTS = [
    f"Document {i}: the river {i % 7} flows past {i % 11} mills before it reaches the sea."
    for i in range(2500)
]


class _CharTokenizer:
    """One token per character (its code point), with EOS id 0."""

    eos_token_id: int | None = EOS

    def __call__(self, text: str, *, add_special_tokens: bool) -> Mapping[str, list[int]]:
        assert not add_special_tokens
        return {"input_ids": [ord(char) for char in text]}


def _resolve(name: str, *, mode: str = "full"):  # type: ignore[no-untyped-def]
    return resolve_config(
        CONFIG_ROOT / "variants" / f"{name}.yaml",
        mode="smoke" if mode == "smoke" else "full",
        config_root=CONFIG_ROOT,
    )


def test_pack_joins_documents_with_eos_and_cuts_full_rows() -> None:
    rows = pack_tokens(["ab", "cde", "fgh"], _CharTokenizer(), seq_len=SEQ_LEN, max_tokens=8)
    expected = [ord("a"), ord("b"), EOS, ord("c"), ord("d"), ord("e"), EOS, ord("f")]
    assert rows.dtype == np.int64
    assert rows.tolist() == [expected[:4], expected[4:]]


def test_pack_stops_reading_once_the_budget_is_met() -> None:
    def texts() -> Iterator[str]:
        yield "abcdefgh"
        raise AssertionError("read past the budget")

    rows = pack_tokens(texts(), _CharTokenizer(), seq_len=SEQ_LEN, max_tokens=SEQ_LEN)
    assert rows.shape == (1, SEQ_LEN)


def test_pack_drops_a_trailing_partial_row() -> None:
    rows = pack_tokens(["abcdef"], _CharTokenizer(), seq_len=SEQ_LEN, max_tokens=100)
    assert rows.shape == (1, SEQ_LEN)


def test_pack_refuses_a_stream_shorter_than_one_row() -> None:
    with pytest.raises(ValueError, match="one 4-token row"):
        pack_tokens(["a"], _CharTokenizer(), seq_len=SEQ_LEN, max_tokens=100)


def test_pack_needs_an_eos_token() -> None:
    tokenizer = _CharTokenizer()
    tokenizer.eos_token_id = None
    with pytest.raises(ValueError, match="eos_token_id"):
        pack_tokens(["abcd"], tokenizer, seq_len=SEQ_LEN, max_tokens=SEQ_LEN)


def test_schedule_warms_up_linearly_then_decays_to_zero() -> None:
    values = [
        lr_multiplier(step, total_steps=TOTAL_STEPS, warmup_steps=WARMUP_STEPS)
        for step in range(TOTAL_STEPS + 1)
    ]
    assert values[0] == pytest.approx(1 / WARMUP_STEPS)
    assert values[WARMUP_STEPS - 1] == pytest.approx(1.0)
    decay = values[WARMUP_STEPS:]
    assert all(later <= earlier for earlier, later in pairwise(decay))
    assert values[TOTAL_STEPS] == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("name", ["qat-3bit-lora", "student-qat-3bit-full"])
def test_the_repo_qat_configs_pass_the_spec_checks(name: str) -> None:
    spec = require_qat_spec(_resolve(name).variant)
    assert spec.corpus == "fineweb_edu"


def test_a_lora_method_without_a_rank_is_refused() -> None:
    variant = _resolve("qat-3bit-lora").variant
    assert variant.qat is not None
    rankless = replace(variant, qat=replace(variant.qat, lora_rank=0))
    with pytest.raises(ValueError, match="no lora_rank"):
        require_qat_spec(rankless)


def test_a_full_method_with_a_rank_is_refused() -> None:
    variant = _resolve("student-qat-3bit-full").variant
    assert variant.qat is not None
    ranked = replace(variant, qat=replace(variant.qat, lora_rank=8))
    with pytest.raises(ValueError, match="sets lora_rank"):
        require_qat_spec(ranked)


def test_a_qat_method_without_a_qat_block_is_refused() -> None:
    variant = _resolve("qat-3bit-lora").variant
    with pytest.raises(ValueError, match="no qat block"):
        require_qat_spec(replace(variant, qat=None))


def test_a_ptq_variant_is_not_a_qat_variant() -> None:
    with pytest.raises(ValueError, match="not a torchao QAT variant"):
        require_qat_spec(_resolve("ptq-3bit-torchao").variant)


@pytest.mark.parametrize(
    ("name", "weights"),
    [("qat-3bit-lora", ADAPTER_FILE), ("student-qat-3bit-full", "config.json")],
)
def test_a_checkpoint_is_complete_with_its_weights_and_metadata(name: str, weights: str) -> None:
    assert completion_markers(_resolve(name).variant) == (weights, METADATA_FILE)


@pytest.mark.parametrize("name", ["qat-3bit-lora", "student-qat-3bit-full"])
def test_producer_returns_early_for_a_complete_checkpoint(tmp_path: Path, name: str) -> None:
    resolved = _resolve(name)
    out = checkpoint_path(resolved.variant, resolved.backend, root=tmp_path, seed=1)
    out.mkdir(parents=True)
    for marker in completion_markers(resolved.variant):
        (out / marker).write_text("{}", encoding="utf-8")

    def no_stream(_corpus: str) -> list[str]:
        raise AssertionError("a complete checkpoint must not train again")

    result = produce_qat(
        resolved.variant,
        resolved.backend,
        checkpoints_root=tmp_path,
        seed=1,
        device="cpu",
        texts=no_stream,
    )
    assert result == out


def test_factors_from_merged_recover_a_low_rank_update_under_merge_noise() -> None:
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    base = {"q": torch.randn(12, 10), "v": torch.randn(8, 10)}
    true = {n: torch.randn(w.shape[0], 2) @ torch.randn(2, 10) * 0.05 for n, w in base.items()}
    # bf16 rounding of the merged sum, as a bf16 LoRA merge leaves it.
    merged = {name: (base[name] + true[name]).bfloat16().float() for name in base}
    factors = factors_from_merged(base, merged, rank=2)
    assert set(factors) == {"q.up", "q.down", "v.up", "v.down"}
    for name, update in true.items():
        assert factors[f"{name}.up"].shape == (update.shape[0], 2)
        assert factors[f"{name}.down"].shape == (2, 10)
        recovered = factors[f"{name}.up"] @ factors[f"{name}.down"]
        noise = (merged[name] - base[name] - update).norm()
        assert (recovered - update).norm() <= noise


def test_factors_from_merged_refuse_mismatched_layers() -> None:
    torch = pytest.importorskip("torch")
    with pytest.raises(ValueError, match="different layers"):
        factors_from_merged({"q": torch.zeros(2, 2)}, {"k": torch.zeros(2, 2)}, rank=1)


@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get("FRONTIER_LIVE_MODELS"),
    reason="live model download; set FRONTIER_LIVE_MODELS=1 to run",
)
@pytest.mark.parametrize("name", ["qat-3bit-lora", "student-qat-3bit-full"])
def test_one_qat_step_on_smollm2_saves_a_checkpoint_the_torchao_provider_serves(
    tmp_path: Path, name: str
) -> None:
    pytest.importorskip("torchao")
    pytest.importorskip("peft")
    from frontier.backends.registry import build_provider  # noqa: PLC0415
    from frontier.eval.prompts import build_prompt  # noqa: PLC0415

    resolved = _resolve(name, mode="smoke")
    one_step = 16 * 1024
    out = produce_qat(
        resolved.variant,
        resolved.backend,
        checkpoints_root=tmp_path,
        seed=0,
        device="cpu",
        max_tokens=one_step,
        texts=lambda _corpus: LIVE_TEXTS,
    )
    metadata = json.loads((out / METADATA_FILE).read_text(encoding="utf-8"))
    assert metadata["steps"] == 1
    assert metadata["tokens_trained"] == one_step
    assert np.isfinite(metadata["final_loss"])

    provider = build_provider(
        resolved.variant,
        resolved.backend,
        device="cpu",
        mode="smoke",
        checkpoints_root=tmp_path,
        seed=0,
    )
    if name == "qat-3bit-lora":
        assert provider.model_id == resolved.variant.model.model_id  # type: ignore[attr-defined]
        assert provider.adapter == out / ADAPTER_FILE  # type: ignore[attr-defined]
        assert not (out / "model.safetensors").exists()
    else:
        assert provider.model_id == str(out)  # type: ignore[attr-defined]
    logits = provider.next_token_logits([build_prompt("What is 2 + 2?", ["3", "4", "5", "6"])])
    assert np.all(np.isfinite(logits))


@pytest.mark.slow
@pytest.mark.skipif(
    not os.environ.get("FRONTIER_LIVE_MODELS"),
    reason="live model download; set FRONTIER_LIVE_MODELS=1 to run",
)
def test_the_served_lora_model_computes_the_trained_function_on_cpu(tmp_path: Path) -> None:
    # CPU only: on CUDA, training quantises bf16 weights and serving fp16 ones (~1% of codes).
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchao")
    pytest.importorskip("peft")
    from frontier.backends.torchao import TorchaoLogitProvider  # noqa: PLC0415
    from frontier.quantize.qat import _lora_factors, _prepared_model, save_adapter  # noqa: PLC0415

    resolved = _resolve("qat-3bit-lora", mode="smoke")
    variant, spec = resolved.variant, require_qat_spec(resolved.variant)
    assert variant.quant is not None
    torch.manual_seed(0)
    trained = _prepared_model(variant, spec, device="cpu", lora=True)
    # A fresh adapter has B = 0; give it weight so the comparison tests something.
    for name, param in trained.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(param, std=0.02)
    trained.eval()
    ids = torch.arange(1, 33).reshape(2, 16)
    with torch.inference_mode():
        expected = trained(input_ids=ids).logits

    def served_logits(adapter: Path | None) -> Any:
        provider = TorchaoLogitProvider(
            model_id=variant.model.model_id,
            device="cpu",
            weight_dtype="int3",
            bit_width=variant.quant.bit_width,
            group_size=variant.quant.group_size,
            adapter=adapter,
        )
        model, _ = provider.loaded_model()
        with torch.inference_mode():
            return model(input_ids=ids).logits

    actual = served_logits(save_adapter(_lora_factors(trained), tmp_path))
    assert not torch.allclose(served_logits(None), expected, atol=1e-3)
    assert torch.allclose(actual, expected, atol=1e-4)
