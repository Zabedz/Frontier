"""Flatten / unflatten and record round-trips for the result store."""

from __future__ import annotations

import math

from rows import nan_row, row_with_latency, sample_row

from frontier.io.serialize import (
    RESULT_COLUMNS,
    RESULT_SCHEMA,
    flatten_record,
    from_record,
    to_record,
    unflatten_record,
)

EXPECTED_COLUMN_COUNT = 47


def test_record_round_trip_with_robustness() -> None:
    row = sample_row(with_robustness=True)
    assert from_record(to_record(row)) == row


def test_record_round_trip_without_robustness() -> None:
    row = sample_row(with_robustness=False)
    restored = from_record(to_record(row))
    assert restored == row
    assert restored.robustness is None


def test_flatten_unflatten_is_inverse() -> None:
    record = to_record(sample_row())
    assert unflatten_record(flatten_record(record)) == record


def test_flatten_unflatten_round_trips_populated_latency() -> None:
    record = to_record(row_with_latency())
    restored = from_record(unflatten_record(flatten_record(record)))
    assert restored == row_with_latency()


def test_flat_row_has_exactly_the_result_columns() -> None:
    flat = flatten_record(to_record(sample_row()))
    assert set(flat) == set(RESULT_COLUMNS)


def test_absent_robustness_flattens_to_nulls_not_nan() -> None:
    flat = flatten_record(to_record(sample_row(with_robustness=False)))
    for name in ("permutation_consistency", "letter_bias", "debias_flip_rate"):
        assert flat[f"robustness.{name}"] is None


def test_nan_record_round_trips_by_isnan() -> None:
    restored = from_record(to_record(nan_row()))
    assert math.isnan(restored.quality.perplexity)
    assert math.isnan(restored.tok_s_per_gb)


def test_result_columns_match_schema() -> None:
    assert len(RESULT_COLUMNS) == EXPECTED_COLUMN_COUNT
    assert list(RESULT_COLUMNS) == RESULT_SCHEMA.names
