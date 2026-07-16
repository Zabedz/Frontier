"""The bitsandbytes weight-dtype mapping, a pure function tested on CPU."""

from __future__ import annotations

import pytest

from frontier.backends.bnb import (
    bnb_config_kwargs,
    is_bnb_dtype,
    resolve_bnb_compute_dtype,
)


def test_is_bnb_dtype_truth_table() -> None:
    assert is_bnb_dtype("nf4")
    assert is_bnb_dtype("int8")
    assert not is_bnb_dtype("fp16")
    assert not is_bnb_dtype("int4")
    assert not is_bnb_dtype("q4_k_m")


def test_nf4_kwargs_are_four_bit_double_quant() -> None:
    kwargs = bnb_config_kwargs("nf4", compute_dtype="bfloat16")
    assert kwargs["load_in_4bit"] is True
    assert kwargs["bnb_4bit_quant_type"] == "nf4"
    assert kwargs["bnb_4bit_use_double_quant"] is True
    assert kwargs["bnb_4bit_compute_dtype"] == "bfloat16"
    assert "load_in_8bit" not in kwargs


def test_int8_kwargs_are_weight_only() -> None:
    kwargs = bnb_config_kwargs("int8", compute_dtype="bfloat16")
    assert kwargs == {"load_in_8bit": True}


def test_bnb_config_kwargs_rejects_non_bnb_dtype() -> None:
    with pytest.raises(ValueError, match="not a bitsandbytes dtype"):
        bnb_config_kwargs("fp16", compute_dtype="bfloat16")


def test_resolve_bnb_compute_dtype_is_bfloat16() -> None:
    assert resolve_bnb_compute_dtype("nf4") == "bfloat16"
    assert resolve_bnb_compute_dtype("int8") == "bfloat16"


def test_resolve_bnb_compute_dtype_rejects_non_bnb_dtype() -> None:
    with pytest.raises(ValueError, match="not a bitsandbytes dtype"):
        resolve_bnb_compute_dtype("fp16")
