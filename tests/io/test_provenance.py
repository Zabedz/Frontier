"""Provenance stamping, git-SHA reading, and hardware identity."""

from __future__ import annotations

import subprocess
from collections.abc import Sequence

from frontier.io.provenance import (
    HardwareInfo,
    hardware_info,
    now_utc_iso,
    read_git_sha,
    stamp_provenance,
)


def test_stamp_provenance_sets_every_field() -> None:
    hardware = HardwareInfo(hardware_id="cpu:arm64", driver_version="none", cuda_version="none")
    expected_seed = 3
    prov = stamp_provenance(
        config_hash="a" * 64,
        model_id="HuggingFaceTB/SmolLM2-135M-Instruct",
        model_revision="main",
        seed=expected_seed,
        hardware=hardware,
        git_sha="deadbeef",
        timestamp="2026-07-13T00:00:00+00:00",
    )
    assert prov.config_hash == "a" * 64
    assert prov.model_id == "HuggingFaceTB/SmolLM2-135M-Instruct"
    assert prov.model_revision == "main"
    assert prov.seed == expected_seed
    assert prov.hardware_id == "cpu:arm64"
    assert prov.driver_version == "none"
    assert prov.cuda_version == "none"
    assert prov.git_sha == "deadbeef"
    assert prov.timestamp == "2026-07-13T00:00:00+00:00"


def test_read_git_sha_clean_tree() -> None:
    def fake_run(command: Sequence[str]) -> str:
        if command[1] == "rev-parse":
            return "abc123\n"
        return ""

    assert read_git_sha(run=fake_run) == "abc123"


def test_read_git_sha_dirty_tree() -> None:
    def fake_run(command: Sequence[str]) -> str:
        if command[1] == "rev-parse":
            return "abc123\n"
        return " M src/frontier/io/store.py\n"

    assert read_git_sha(run=fake_run) == "abc123-dirty"


def test_read_git_sha_no_repo_returns_nogit() -> None:
    def missing_git(command: Sequence[str]) -> str:  # noqa: ARG001
        raise FileNotFoundError("git")

    def nonzero_exit(command: Sequence[str]) -> str:
        raise subprocess.CalledProcessError(128, command)

    assert read_git_sha(run=missing_git) == "nogit"
    assert read_git_sha(run=nonzero_exit) == "nogit"


def test_hardware_info_cpu() -> None:
    info = hardware_info(device="cpu")
    assert info.driver_version == "none"
    assert info.cuda_version == "none"
    assert info.hardware_id.startswith("cpu:")


def test_now_utc_iso_is_offset_aware() -> None:
    stamp = now_utc_iso()
    assert stamp.endswith("+00:00")
