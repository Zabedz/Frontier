"""Provenance stamping: git SHA, hardware identity, timestamp, and the assembled record.

Values that vary with the machine or the moment come in through overridable seams, so
``stamp_provenance`` is deterministic and the tests pin exact fields. The GPU branch of
``hardware_info`` is exercised on the pod.
"""

from __future__ import annotations

import platform
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime

from frontier.schema import Provenance

GitRunner = Callable[[Sequence[str]], str]


@dataclass(frozen=True, slots=True)
class HardwareInfo:
    """The hardware / driver / CUDA triple for a row's provenance."""

    hardware_id: str
    driver_version: str
    cuda_version: str


def _run_git(command: Sequence[str]) -> str:
    result = subprocess.run(list(command), capture_output=True, text=True, check=True)
    return result.stdout


def read_git_sha(*, run: GitRunner | None = None) -> str:
    """The current commit, with a ``"-dirty"`` suffix when the tree has changes.

    A checkout with no ``.git`` returns ``"nogit"``, so a row still produces.
    """
    execute = run or _run_git
    try:
        head = execute(["git", "rev-parse", "HEAD"]).strip()
        status = execute(["git", "status", "--porcelain"])
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "nogit"
    return f"{head}-dirty" if status.strip() else head


def hardware_info(*, device: str) -> HardwareInfo:
    """Hardware identity for the row.

    On ``cpu`` the driver and CUDA versions are the ``"none"`` sentinel, since the schema
    fields are non-optional ``str``.
    """
    if device == "cpu":
        return HardwareInfo(
            hardware_id=f"cpu:{platform.machine()}",
            driver_version="none",
            cuda_version="none",
        )
    import torch  # noqa: PLC0415  # pragma: no cover

    return HardwareInfo(  # pragma: no cover
        hardware_id=torch.cuda.get_device_name(0),
        driver_version=_nvidia_driver(),
        cuda_version=torch.version.cuda or "unknown",
    )


def _nvidia_driver() -> str:  # pragma: no cover
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    lines = result.stdout.strip().splitlines()
    return lines[0].strip() if lines else "unknown"


def now_utc_iso() -> str:
    """Current UTC time as an ISO-8601 string."""
    return datetime.now(UTC).isoformat()


def stamp_provenance(
    *,
    config_hash: str,
    model_id: str,
    model_revision: str,
    seed: int,
    hardware: HardwareInfo,
    git_sha: str,
    timestamp: str,
) -> Provenance:
    """Assemble a ``schema.Provenance`` from fully supplied inputs."""
    return Provenance(
        git_sha=git_sha,
        config_hash=config_hash,
        model_id=model_id,
        model_revision=model_revision,
        hardware_id=hardware.hardware_id,
        driver_version=hardware.driver_version,
        cuda_version=hardware.cuda_version,
        seed=seed,
        timestamp=timestamp,
    )
