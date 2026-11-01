"""The distillation arm live on SmolLM2-135M: the cache at the smoke cap with its fidelity
gate, one step per arm, and the hf provider serving each trained student."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
from tiny import ARMS, CONFIG_ROOT, ONE_STEP_TOKENS

from frontier.backends.hf import HFLogitProvider
from frontier.backends.registry import build_provider
from frontier.distill.build import ensure_cache
from frontier.distill.fidelity import read_fidelity
from frontier.distill.paths import STUDENT_METADATA_FILE
from frontier.distill.train import produce_distill
from frontier.eval.prompts import build_prompt
from frontier.pipeline.config import ResolvedConfig, resolve_config

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not os.environ.get("FRONTIER_LIVE_MODELS"),
        reason="live model download; set FRONTIER_LIVE_MODELS=1 to run",
    ),
]

# The QAT live test's sentences: enough SmolLM2 tokens for 16 training and 4 held-out rows.
LIVE_TEXTS = [
    f"Document {i}: the river {i % 7} flows past {i % 11} mills before it reaches the sea."
    for i in range(2500)
]


def _live_texts(_corpus: str) -> list[str]:
    return LIVE_TEXTS


def _resolve(name: str) -> ResolvedConfig:
    return resolve_config(
        CONFIG_ROOT / "variants" / f"{name}.yaml", mode="smoke", config_root=CONFIG_ROOT
    )


def test_the_smoke_cache_passes_fidelity_and_peaks_at_the_next_token(tmp_path: Path) -> None:
    pytest.importorskip("transformers")
    cache = ensure_cache(
        _resolve("distill-softlabel-topk").variant,
        checkpoints_root=tmp_path,
        device="cpu",
        max_tokens=ONE_STEP_TOKENS,
        texts=_live_texts,
    )
    report = read_fidelity(cache.path)
    assert report.passed
    agreement = report.next_token_agreement
    assert agreement[1] > max(agreement[0], agreement[2])


def test_each_arm_trains_and_its_student_is_served(tmp_path: Path) -> None:
    pytest.importorskip("transformers")
    prompt = build_prompt("What is 2 + 2?", ["3", "4", "5", "6"])
    for name in ARMS:
        resolved = _resolve(name)
        out = produce_distill(
            resolved.variant,
            checkpoints_root=tmp_path,
            seed=0,
            device="cpu",
            max_tokens=ONE_STEP_TOKENS,
            texts=_live_texts,
        )
        metadata = json.loads((out / STUDENT_METADATA_FILE).read_text(encoding="utf-8"))
        assert metadata["steps"] == 1
        assert np.isfinite(metadata["final_loss"])
        provider = build_provider(
            resolved.variant,
            resolved.backend,
            device="cpu",
            mode="smoke",
            checkpoints_root=tmp_path,
            seed=0,
        )
        assert isinstance(provider, HFLogitProvider)
        assert provider.model_id == str(out)
        assert np.all(np.isfinite(provider.next_token_logits([prompt])))
