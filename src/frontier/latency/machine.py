"""nvidia-smi capture, CSV parse, the clock-lock probe, and the no-GPU degradation.

A latency number is only defensible with the clock and thermal state it was measured
under. On RunPod the clocks usually cannot be locked (the container lacks the
privilege), so the fallback the methodology requires is to log ``clocks.sm``,
``clocks.mem``, ``temperature.gpu``, and ``power.draw`` around every measurement and
flag a run whose clocks drifted. Every path here degrades to a well-defined no-GPU
``MachineState`` when nvidia-smi is absent, which is the row a laptop carries.

The subprocess runner is injected the way ``io.provenance`` injects its git runner, so
the parse and the failure branches are unit-tested without a GPU.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Protocol

from frontier.schema import MachineState

_QUERY_FIELDS = ("clocks.sm", "clocks.mem", "temperature.gpu", "power.draw")
_QUERY_GPU_ARG = "--query-gpu=" + ",".join(_QUERY_FIELDS)
_CSV_FORMAT = "--format=csv,noheader,nounits"
_NOT_AVAILABLE = frozenset({"[N/A]", "N/A", ""})
DEFAULT_DRIFT_TOL_MHZ = 15

SmiRunner = Callable[[Sequence[str]], str]


@dataclass(frozen=True, slots=True)
class ClockReading:
    """One nvidia-smi sample. ``present`` is ``False`` when no GPU was queryable."""

    sm_mhz: int
    mem_mhz: int
    temp_c: int
    power_w: float
    present: bool

    @classmethod
    def no_gpu(cls) -> ClockReading:
        return cls(0, 0, 0, 0.0, present=False)


class MachineProbe(Protocol):
    """Captures a live ``ClockReading`` around a measurement block."""

    def capture(self) -> ClockReading: ...


class NvidiaSmiProbe:
    """The real ``MachineProbe``: one ``nvidia-smi`` query, ``no_gpu`` off a laptop."""

    def capture(self) -> ClockReading:
        return query_nvidia_smi()


def parse_nvidia_smi_csv(text: str) -> ClockReading:
    """Parse one line of ``--query-gpu=... --format=csv,noheader,nounits``.

    Takes the first non-empty line (single-GPU target; extra GPU lines are ignored) and
    splits on commas. Clocks and temperature go through ``int(float(x))`` so ``"1500.00"``
    parses; power through ``float``. A field that is ``"[N/A]"`` or ``"N/A"`` (power can be
    N/A on some cards) becomes ``0`` for that field while ``present`` stays ``True``: the
    binary ran and one field was unavailable. A line with no parseable number in any field
    returns ``no_gpu``.
    """
    line = next((candidate for candidate in text.splitlines() if candidate.strip()), "")
    fields = [field.strip() for field in line.split(",")]
    if len(fields) < len(_QUERY_FIELDS):
        return ClockReading.no_gpu()
    sm, sm_ok = _parse_int(fields[0])
    mem, mem_ok = _parse_int(fields[1])
    temp, temp_ok = _parse_int(fields[2])
    power, power_ok = _parse_float(fields[3])
    if not (sm_ok or mem_ok or temp_ok or power_ok):
        return ClockReading.no_gpu()
    return ClockReading(sm_mhz=sm, mem_mhz=mem, temp_c=temp, power_w=power, present=True)


def query_nvidia_smi(*, run: SmiRunner | None = None) -> ClockReading:
    """Shell out and parse, degrading to ``no_gpu`` when nvidia-smi is absent or errors.

    The default runner wraps ``subprocess.run(..., check=True, capture_output=True,
    text=True)``. ``FileNotFoundError`` (no binary, the laptop case) and
    ``CalledProcessError`` are caught and return ``no_gpu``; nothing else is swallowed.
    """
    execute = run or _run_smi
    try:
        output = execute(["nvidia-smi", _QUERY_GPU_ARG, _CSV_FORMAT])
    except (FileNotFoundError, subprocess.CalledProcessError):
        return ClockReading.no_gpu()
    return parse_nvidia_smi_csv(output)


def probe_clock_lock(*, run: SmiRunner | None = None) -> bool:
    """Try to enable persistence mode and lock the GPU and memory clocks to their max.

    Returns ``True`` only if every step exits 0. On RunPod the expected result is
    ``False``: ``-pm 1`` already fails with "Insufficient Permissions" in an
    unprivileged container, and a missing binary is likewise ``False``. The caught
    failures are ``FileNotFoundError`` and ``CalledProcessError`` only.

    Side effect: the success path mutates shared host state (persistence mode on, SM
    and memory clocks pinned), and that change outlives the process. If this ever
    returns ``True`` on a pod, record it in SSH_CHANGELOG.md and reset the clocks
    afterwards (``nvidia-smi -rgc -rmc``, ``nvidia-smi -pm 0``).
    """
    execute = run or _run_smi
    try:
        execute(["nvidia-smi", "-pm", "1"])
        max_sm, max_mem = _query_max_clocks(execute)  # pragma: no cover
        execute(["nvidia-smi", f"--lock-gpu-clocks=0,{max_sm}"])  # pragma: no cover
        execute(["nvidia-smi", f"--lock-memory-clocks=0,{max_mem}"])  # pragma: no cover
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False
    return True  # pragma: no cover


def to_machine_state(
    before: ClockReading,
    after: ClockReading,
    *,
    clocks_locked: bool,
    drift_tol_mhz: int = DEFAULT_DRIFT_TOL_MHZ,
) -> MachineState:
    """Assemble a ``schema.MachineState`` from a before/after clock pair.

    The reported clocks, temperature, and power come from ``after`` (steady state at the
    end of the batch's trials). ``clock_drift_flag`` is ``True`` when the SM or memory
    clock moved by more than ``drift_tol_mhz`` between the two readings, so a run whose
    clocks shifted is flagged rather than silently trusted: two variants are only
    comparable under the same logged clock state. If either reading is not ``present`` the
    result is the well-defined no-GPU state that a laptop row carries.
    """
    if not (before.present and after.present):
        return MachineState(0, 0, 0, 0.0, clocks_locked=False, clock_drift_flag=False)
    drift = (
        abs(after.sm_mhz - before.sm_mhz) > drift_tol_mhz
        or abs(after.mem_mhz - before.mem_mhz) > drift_tol_mhz
    )
    return MachineState(
        gpu_clock_sm_mhz=after.sm_mhz,
        gpu_clock_mem_mhz=after.mem_mhz,
        gpu_temp_c=after.temp_c,
        power_w=after.power_w,
        clocks_locked=clocks_locked,
        clock_drift_flag=drift,
    )


def _run_smi(command: Sequence[str]) -> str:
    result = subprocess.run(list(command), check=True, capture_output=True, text=True)
    return result.stdout


def _query_max_clocks(execute: SmiRunner) -> tuple[int, int]:  # pragma: no cover
    output = execute(["nvidia-smi", "--query-gpu=clocks.max.sm,clocks.max.mem", _CSV_FORMAT])
    fields = [field.strip() for field in output.splitlines()[0].split(",")]
    sm, _ = _parse_int(fields[0])
    mem, _ = _parse_int(fields[1])
    return sm, mem


def _parse_int(raw: str) -> tuple[int, bool]:
    if raw in _NOT_AVAILABLE:
        return 0, False
    try:
        return int(float(raw)), True
    except ValueError:
        return 0, False


def _parse_float(raw: str) -> tuple[float, bool]:
    if raw in _NOT_AVAILABLE:
        return 0.0, False
    try:
        return float(raw), True
    except ValueError:
        return 0.0, False
