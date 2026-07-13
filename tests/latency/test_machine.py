"""nvidia-smi parse and the no-GPU degradation, all through injected runners."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence

from frontier.latency.machine import (
    ClockReading,
    parse_nvidia_smi_csv,
    probe_clock_lock,
    query_nvidia_smi,
    to_machine_state,
)

SM_MHZ = 1500
MEM_MHZ = 6000
TEMP_C = 61
POWER_W = 118.5
DRIFTED_SM_MHZ = 1400


def test_parse_bare_numbers() -> None:
    reading = parse_nvidia_smi_csv(f"{SM_MHZ}, {MEM_MHZ}, {TEMP_C}, {POWER_W}")
    assert reading == ClockReading(SM_MHZ, MEM_MHZ, TEMP_C, POWER_W, present=True)


def test_parse_float_clock_via_int_of_float() -> None:
    reading = parse_nvidia_smi_csv(f"{SM_MHZ}.00, {MEM_MHZ}.00, {TEMP_C}, {POWER_W}")
    assert reading.sm_mhz == SM_MHZ
    assert reading.mem_mhz == MEM_MHZ


def test_parse_na_power_keeps_present() -> None:
    reading = parse_nvidia_smi_csv(f"{SM_MHZ}, {MEM_MHZ}, {TEMP_C}, [N/A]")
    assert reading.present is True
    assert reading.power_w == 0.0
    assert reading.sm_mhz == SM_MHZ


def test_parse_junk_line_is_no_gpu() -> None:
    assert parse_nvidia_smi_csv("No devices were found") == ClockReading.no_gpu()


def test_query_degrades_when_binary_absent() -> None:
    def missing(command: Sequence[str]) -> str:
        raise FileNotFoundError(command[0])

    reading = query_nvidia_smi(run=missing)
    assert reading.present is False
    assert reading == ClockReading.no_gpu()


def test_query_degrades_on_nonzero_exit() -> None:
    def failing(command: Sequence[str]) -> str:
        raise subprocess.CalledProcessError(1, list(command))

    assert query_nvidia_smi(run=failing) == ClockReading.no_gpu()


def test_to_machine_state_flags_no_drift_when_clocks_hold() -> None:
    reading = ClockReading(SM_MHZ, MEM_MHZ, TEMP_C, POWER_W, present=True)
    state = to_machine_state(reading, reading, clocks_locked=True)
    assert state.clock_drift_flag is False
    assert state.clocks_locked is True
    assert state.gpu_clock_sm_mhz == SM_MHZ


def test_to_machine_state_flags_drift_when_sm_moves() -> None:
    before = ClockReading(SM_MHZ, MEM_MHZ, TEMP_C, POWER_W, present=True)
    after = ClockReading(DRIFTED_SM_MHZ, MEM_MHZ, TEMP_C + 1, POWER_W + 1.5, present=True)
    state = to_machine_state(before, after, clocks_locked=False)
    assert state.clock_drift_flag is True
    # Reported clocks are the steady-state "after" reading.
    assert state.gpu_clock_sm_mhz == DRIFTED_SM_MHZ


def test_to_machine_state_absent_reading_is_zero_state() -> None:
    present = ClockReading(SM_MHZ, MEM_MHZ, TEMP_C, POWER_W, present=True)
    state = to_machine_state(ClockReading.no_gpu(), present, clocks_locked=True)
    assert state.gpu_clock_sm_mhz == 0
    assert state.power_w == 0.0
    assert state.clocks_locked is False
    assert state.clock_drift_flag is False


def test_probe_clock_lock_false_when_binary_absent() -> None:
    def missing(command: Sequence[str]) -> str:
        raise FileNotFoundError(command[0])

    assert probe_clock_lock(run=missing) is False


def test_probe_clock_lock_false_on_permission_denied() -> None:
    def denied(command: Sequence[str]) -> str:
        raise subprocess.CalledProcessError(4, list(command), stderr="Insufficient Permissions")

    assert probe_clock_lock(run=denied) is False
