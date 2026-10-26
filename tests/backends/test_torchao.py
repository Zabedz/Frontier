"""The torchao backend: bit-width guard, the shared intx config, the lm_head filter, and a
live SmolLM2-135M CPU check that 3-bit PTQ moves the logits."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import pytest

from frontier.backends.torchao import (
    MAX_BITS,
    MIN_BITS,
    TorchaoLogitProvider,
    attach_adapter,
    intx_weight_only_config,
    quantizes,
    require_bit_width,
)

SMOL = "HuggingFaceTB/SmolLM2-135M-Instruct"
BIT_WIDTH = 3
GROUP_SIZE = 32
# Cosine between the fp32 and 3-bit answer-position logits; int3 on 135M sits near 0.9.
MAX_COSINE = 0.999

live = pytest.mark.skipif(
    not os.environ.get("FRONTIER_LIVE_MODELS"),
    reason="live model download; set FRONTIER_LIVE_MODELS=1 to run",
)


@pytest.mark.parametrize("bits", [MIN_BITS - 1, MAX_BITS + 1])
def test_bit_width_outside_torch_intx_is_refused(bits: int) -> None:
    with pytest.raises(ValueError, match="bit width"):
        require_bit_width(bits)


def test_the_provider_refuses_a_bad_bit_width_before_loading() -> None:
    with pytest.raises(ValueError, match="bit width"):
        TorchaoLogitProvider(
            model_id=SMOL, device="cpu", weight_dtype="int8", bit_width=8, group_size=32
        )


def test_config_carries_the_bit_width_and_group() -> None:
    torch = pytest.importorskip("torch")
    pytest.importorskip("torchao")
    config = intx_weight_only_config(BIT_WIDTH, GROUP_SIZE)
    assert config.weight_dtype is torch.int3
    assert config.granularity.group_size == GROUP_SIZE


def test_filter_takes_linears_and_skips_the_output_head() -> None:
    torch = pytest.importorskip("torch")
    linear = torch.nn.Linear(4, 4)
    assert quantizes(linear, "model.layers.0.mlp.down_proj")
    assert not quantizes(linear, "lm_head")
    assert not quantizes(torch.nn.Embedding(4, 4), "model.embed_tokens")


def test_cpu_computes_in_fp32_and_cuda_in_the_fp16_of_its_reference() -> None:
    def provider(device: str) -> TorchaoLogitProvider:
        return TorchaoLogitProvider(
            model_id=SMOL, device=device, weight_dtype="int3", bit_width=3, group_size=32
        )

    assert provider("cpu")._compute_dtype_name() == "float32"
    assert provider("cuda")._compute_dtype_name() == "float16"


def _two_layer_net() -> Any:
    torch = pytest.importorskip("torch")
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Linear(6, 5), torch.nn.ReLU(), torch.nn.Linear(5, 4))


def test_an_attached_adapter_adds_its_low_rank_product_to_the_layer_output() -> None:
    torch = pytest.importorskip("torch")
    net = _two_layer_net()
    x = torch.randn(3, 6)
    with torch.no_grad():
        hidden = torch.relu(net[0](x))
        expected = net[2](hidden)
    up, down = torch.randn(4, 2), torch.randn(2, 5)
    assert attach_adapter(net, {"2.up": up, "2.down": down}, dtype=torch.float32) == 1
    with torch.no_grad():
        assert torch.allclose(net(x), expected + hidden @ down.T @ up.T, atol=1e-6)
        assert torch.allclose(net[0](x), torch.nn.functional.linear(x, net[0].weight, net[0].bias))


@pytest.mark.parametrize(
    ("factors", "message"),
    [
        ({}, "no factors"),
        ({"1.up": (5, 1), "1.down": (1, 5)}, "not a linear layer"),
        ({"2.up": (4, 1)}, "needs both"),
        ({"2.up": (4, 2), "2.down": (3, 5)}, "for a 4 x 5 layer"),
        ({"2.up": (4, 1), "2.down": (1, 6)}, "for a 4 x 5 layer"),
        ({"2.up": (4,), "2.down": (1, 5)}, "for a 4 x 5 layer"),
    ],
)
def test_a_malformed_adapter_is_refused_and_nothing_is_attached(
    factors: dict[str, tuple[int, ...]], message: str
) -> None:
    torch = pytest.importorskip("torch")
    net = _two_layer_net()
    tensors = {key: torch.ones(shape) for key, shape in factors.items()}
    with pytest.raises(ValueError, match=message):
        attach_adapter(net, tensors, dtype=torch.float32)
    assert not any(hasattr(layer, "adapter_up") for layer in net)


def test_a_second_adapter_on_the_same_layer_is_refused() -> None:
    torch = pytest.importorskip("torch")
    net = _two_layer_net()
    factors = {"2.up": torch.ones(4, 1), "2.down": torch.ones(1, 5)}
    attach_adapter(net, factors, dtype=torch.float32)
    with pytest.raises(ValueError, match="already carries an adapter"):
        attach_adapter(net, factors, dtype=torch.float32)


def test_adapter_factors_are_stored_in_the_compute_dtype() -> None:
    torch = pytest.importorskip("torch")
    net = _two_layer_net().to(torch.float64)
    attach_adapter(net, {"2.up": torch.ones(4, 1), "2.down": torch.ones(1, 5)}, dtype=torch.float64)
    assert net[2].adapter_up.dtype == net[2].adapter_down.dtype == torch.float64
    assert net(torch.ones(1, 6, dtype=torch.float64)).dtype == torch.float64


@pytest.mark.slow
@live
def test_three_bit_ptq_quantises_every_linear_but_the_head_and_moves_the_logits() -> None:
    pytest.importorskip("torchao")
    from frontier.backends.hf import HFLogitProvider  # noqa: PLC0415
    from frontier.eval.prompts import build_prompt  # noqa: PLC0415

    prompts = [build_prompt("What is 2 + 2?", ["3", "4", "5", "6"])]
    reference = HFLogitProvider(model_id=SMOL, device="cpu", weight_dtype="fp16")
    quantised = TorchaoLogitProvider(
        model_id=SMOL, device="cpu", weight_dtype="int3", bit_width=3, group_size=32
    )
    ref_logits = reference.next_token_logits(prompts)[0]
    q_logits = quantised.next_token_logits(prompts)[0]
    assert np.all(np.isfinite(q_logits))
    cosine = float(ref_logits @ q_logits / (np.linalg.norm(ref_logits) * np.linalg.norm(q_logits)))
    assert cosine < MAX_COSINE
    model, _ = quantised.loaded_model()
    assert type(model.model.layers[0].mlp.down_proj.weight).__name__ != "Parameter"
    assert type(model.lm_head.weight).__name__ == "Parameter"
    assert quantised.backend_version.startswith("torchao 0.")
