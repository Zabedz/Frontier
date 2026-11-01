"""The offline teacher cache: its format, the build primitives, and the memmap read side.

Position ``t`` of a row holds the teacher's top-k and rest for token ``t + 1``.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from frontier.schema import DistillMode, TrainingCorpus
from frontier.training.corpus import TRAINING_CORPORA, TokenRows

CACHE_FORMAT = 1
TEACHER_BATCH_ROWS = 4
FLUSH_EVERY_ROWS = 64  # a multiple of TEACHER_BATCH_ROWS; a crash loses at most this many rows
META_FILE = "cache.json"
INPUT_IDS_FILE = "input_ids.npy"
TOPK_IDS_FILE = "topk_ids.npy"
TOPK_PROBS_FILE = "topk_probs.npy"
REST_FILE = "rest.npy"
VERSIONED_PACKAGES = ("torch", "transformers", "numpy", "datasets", "tokenizers")
_FINGERPRINT_BLOCK_ROWS = 4096
_REST_PERCENTILES = {"p50": 50.0, "p99": 99.0}

Forward = Callable[[Any], Any]  # int64 ids (B, L) on the device -> logits (B, L, V)


@dataclass(frozen=True, slots=True)
class CacheRequest:
    """What a variant needs from the cache; compared against ``cache.json`` on reuse."""

    teacher_id: str
    teacher_revision: str
    corpus: TrainingCorpus
    seq_len: int
    top_k: int
    train_rows: int
    heldout_rows: int  # packed after the training rows; no trainer reads them
    budget_tokens: int
    capped: bool  # the budget is below the variant's train_tokens


@dataclass(frozen=True, slots=True)
class TeacherFacts:
    """What the build records about the teacher it loaded and how it ran it."""

    vocab_size: int
    eos_token_id: int
    tokenizer_fingerprint: str
    commit: str | None  # config._commit_hash of the loaded snapshot; None for a local dir
    dtype: str
    device: str  # "cpu" or torch.cuda.get_device_name()
    deterministic: bool  # torch.use_deterministic_algorithms was on for the forward
    cublas_workspace_config: str | None  # CUBLAS_WORKSPACE_CONFIG at the build


@dataclass(frozen=True, slots=True)
class CacheMeta:
    """``cache.json``; the build moves ``rows_done`` and sets ``complete`` last."""

    format: int
    teacher_id: str
    teacher_revision: str
    teacher_commit: str | None
    corpus: str
    corpus_source: list[str]  # the TRAINING_CORPORA entry: hf path, config, split, text field
    seq_len: int
    top_k: int
    n_rows: int  # training rows plus held-out rows, the arrays' first axis
    heldout_rows: int
    budget_tokens: int
    capped: bool  # built under --max-tokens or the smoke cap
    vocab_size: int
    eos_token_id: int
    tokenizer_fingerprint: str
    rows_fingerprint: str
    teacher_dtype: str
    device: str
    deterministic: bool
    cublas_workspace_config: str | None
    versions: dict[str, str]
    batch_rows: int
    flush_rows: int
    rows_done: int
    complete: bool
    wall_seconds: float  # teacher forward time, summed over build sessions
    rest_summary: dict[str, float] | None  # mean, p50, p99, max of every rest; set at completion


class CacheMismatchError(ValueError):
    """An existing cache was built for another request, or by another teacher setup."""


class VocabTokenizer(Protocol):
    """What the tokenizer fingerprint reads."""

    eos_token_id: int | None

    def get_vocab(self) -> dict[str, int]: ...


@dataclass(frozen=True, slots=True)
class CacheBatch:
    """One micro-batch of cached rows, as copies.

    The teacher arrays are ``None`` in corpus mode; hard mode carries column 0 only, shape
    ``(B, L - 1, 1)``.
    """

    input_ids: npt.NDArray[np.int64]
    topk_ids: npt.NDArray[np.int64] | None
    topk_probs: npt.NDArray[np.float32] | None
    rest: npt.NDArray[np.float32] | None


class TeacherCache:
    """A complete cache, opened read-only as memmaps; the rows fingerprint is checked on open.

    The last ``meta.heldout_rows`` rows are held out, and no trainer reads them.
    """

    def __init__(self, path: Path) -> None:
        meta = read_meta(path)
        if not meta.complete:
            raise ValueError(
                f"teacher cache {path} is incomplete ({meta.rows_done} of {meta.n_rows} rows "
                f"filled); finish it with `frontier-quantize cache`"
            )
        input_ids: npt.NDArray[np.uint32] = np.load(path / INPUT_IDS_FILE, mmap_mode="r")
        if rows_fingerprint(input_ids) != meta.rows_fingerprint:
            raise ValueError(
                f"{path / INPUT_IDS_FILE} does not match the rows fingerprint in {META_FILE}"
            )
        self.path = path
        self._meta = meta
        self._input_ids = input_ids
        self._topk_ids: npt.NDArray[np.uint32] = np.load(path / TOPK_IDS_FILE, mmap_mode="r")
        self._topk_probs: npt.NDArray[np.float16] = np.load(path / TOPK_PROBS_FILE, mmap_mode="r")
        self._rest: npt.NDArray[np.float32] = np.load(path / REST_FILE, mmap_mode="r")

    @property
    def meta(self) -> CacheMeta:
        return self._meta

    @property
    def training(self) -> range:
        """The rows a trainer may read: every row before the held-out block."""
        return range(self._meta.n_rows - self._meta.heldout_rows)

    @property
    def heldout(self) -> range:
        """The rows after the training rows."""
        return range(self._meta.n_rows - self._meta.heldout_rows, self._meta.n_rows)

    def batch(self, rows: npt.NDArray[np.int64], mode: DistillMode) -> CacheBatch:
        """Copies of ``rows`` with the targets ``mode`` reads."""
        input_ids = self._input_ids[rows].astype(np.int64)
        if mode == "corpus_label":
            return CacheBatch(input_ids, None, None, None)
        if mode == "hard_label":
            return CacheBatch(input_ids, self._topk_ids[rows, :, :1].astype(np.int64), None, None)
        return CacheBatch(
            input_ids,
            self._topk_ids[rows].astype(np.int64),
            self._topk_probs[rows].astype(np.float32),
            self._rest[rows].astype(np.float32),
        )


def tokenizer_fingerprint(tokenizer: VocabTokenizer) -> str:
    """sha256 over the sorted vocabulary and the eos id."""
    payload = json.dumps(sorted(tokenizer.get_vocab().items()))
    return hashlib.sha256(f"{payload}|eos={tokenizer.eos_token_id}".encode()).hexdigest()


def rows_fingerprint(rows: npt.NDArray[np.uint32]) -> str:
    """sha256 over ``"<n_rows>x<seq_len>:"`` and the little-endian token ids."""
    digest = hashlib.sha256(f"{rows.shape[0]}x{rows.shape[1]}:".encode())
    for start in range(0, rows.shape[0], _FINGERPRINT_BLOCK_ROWS):
        block = rows[start : start + _FINGERPRINT_BLOCK_ROWS]
        digest.update(np.ascontiguousarray(block, dtype="<u4").tobytes())
    return digest.hexdigest()


def library_versions() -> dict[str, str]:
    """The installed versions of the packages the targets depend on."""
    found: dict[str, str] = {}
    for package in VERSIONED_PACKAGES:
        try:
            found[package] = version(package)
        except PackageNotFoundError:
            continue
    return found


def device_name(device: str) -> str:
    """``"cpu"``, or the CUDA card's name."""
    if device == "cpu":
        return "cpu"
    import torch  # noqa: PLC0415

    return str(torch.cuda.get_device_name(device))


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write through a synced temporary file, so a crash leaves the previous file whole."""
    staging = path.with_name(f"{path.name}.tmp")
    with staging.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    staging.replace(path)


def read_meta(out: Path) -> CacheMeta:
    """The ``cache.json`` of the cache at ``out``; raises ``ValueError`` for another layout."""
    path = out / META_FILE
    try:
        return CacheMeta(**json.loads(path.read_text(encoding="utf-8")))
    except TypeError as error:
        raise ValueError(
            f"{path} is not a format-{CACHE_FORMAT} cache record ({error}); rebuild the cache"
        ) from error


def _write_meta(out: Path, meta: CacheMeta) -> None:
    write_json(out / META_FILE, asdict(meta))


def teacher_targets(logits: Any, top_k: int, *, row: int | None = None) -> tuple[Any, Any, Any]:
    """(ids int64, probs float16, rest float32) for logits (..., V), sorted descending.

    ``rest`` is the mass outside the k ids, from the complement's logsumexp, so a confident
    teacher keeps a rest that ``1 - sum(probs)`` would round to zero. Raises
    ``FloatingPointError`` on a non-finite logit, naming ``row`` when given, and
    ``ValueError`` when ``top_k`` leaves no complement.
    """
    import torch  # noqa: PLC0415

    z = logits.float()
    finite = torch.isfinite(z)
    if not bool(finite.all()):
        index = tuple(int(i) for i in torch.nonzero(~finite)[0])
        where = "" if row is None else f"row {row} "
        raise FloatingPointError(
            f"teacher logit {float(z[index])} at {where}index {index} is not finite"
        )
    vocab = int(z.shape[-1])
    if top_k >= vocab:
        raise ValueError(f"top_k {top_k} leaves no rest bucket in a {vocab}-entry vocabulary")
    lse = torch.logsumexp(z, dim=-1, keepdim=True)
    values, ids = torch.topk(z, top_k, dim=-1, sorted=True)
    probs = torch.exp(values - lse).to(torch.float16)
    outside = torch.logsumexp(z.scatter(-1, ids, float("-inf")), dim=-1)
    return ids, probs, torch.exp(outside - lse.squeeze(-1))


def start_cache(
    out: Path,
    rows: TokenRows,
    request: CacheRequest,
    facts: TeacherFacts,
    *,
    batch_rows: int = TEACHER_BATCH_ROWS,
    flush_rows: int = FLUSH_EVERY_ROWS,
) -> CacheMeta:
    """Write the packed rows, preallocate the target arrays, and write ``cache.json``.

    Raises ``ValueError`` when the rows do not match the request, hold an id outside the
    teacher's vocabulary, or ``top_k`` leaves no rest bucket.
    """
    if flush_rows % batch_rows:
        raise ValueError(f"flush_rows {flush_rows} is not a multiple of batch_rows {batch_rows}")
    expected = (request.train_rows + request.heldout_rows, request.seq_len)
    if rows.shape != expected:
        raise ValueError(f"packed rows have shape {rows.shape}; the request needs {expected}")
    if request.top_k >= facts.vocab_size:
        raise ValueError(
            f"top_k {request.top_k} leaves no rest bucket in the teacher's "
            f"{facts.vocab_size}-entry vocabulary"
        )
    if int(rows.min()) < 0 or int(rows.max()) >= facts.vocab_size:
        raise ValueError(
            f"packed ids span [{rows.min()}, {rows.max()}], outside the teacher's "
            f"{facts.vocab_size}-entry vocabulary"
        )
    out.mkdir(parents=True, exist_ok=True)
    tokens = rows.astype(np.uint32)
    staging = out / f"{INPUT_IDS_FILE}.tmp"
    with staging.open("wb") as handle:
        np.save(handle, tokens)
        handle.flush()
        os.fsync(handle.fileno())
    staging.replace(out / INPUT_IDS_FILE)
    n_rows, seq_len = tokens.shape
    positions = seq_len - 1
    for name, dtype, shape in (
        (TOPK_IDS_FILE, np.uint32, (n_rows, positions, request.top_k)),
        (TOPK_PROBS_FILE, np.float16, (n_rows, positions, request.top_k)),
        (REST_FILE, np.float32, (n_rows, positions)),
    ):
        target = np.lib.format.open_memmap(  # type: ignore[no-untyped-call]
            out / name, mode="w+", dtype=dtype, shape=shape
        )
        target.flush()
    meta = CacheMeta(
        format=CACHE_FORMAT,
        teacher_id=request.teacher_id,
        teacher_revision=request.teacher_revision,
        corpus=request.corpus,
        corpus_source=list(TRAINING_CORPORA[request.corpus]),
        seq_len=seq_len,
        top_k=request.top_k,
        n_rows=n_rows,
        heldout_rows=request.heldout_rows,
        budget_tokens=request.budget_tokens,
        capped=request.capped,
        rows_fingerprint=rows_fingerprint(tokens),
        batch_rows=batch_rows,
        flush_rows=flush_rows,
        rows_done=0,
        complete=False,
        wall_seconds=0.0,
        rest_summary=None,
        **_session(facts),
    )
    _write_meta(out, meta)
    return meta


def fill_targets(
    out: Path,
    forward: Forward,
    *,
    facts: TeacherFacts,
    device: str,
    stop_after: int | None = None,
) -> CacheMeta:
    """Run the teacher over the rows past the watermark and store its targets.

    The memmaps are flushed and the watermark moved every ``flush_rows`` rows, so a resume
    redoes at most that many, over the same batch boundaries. ``stop_after`` ends the fill
    once that many rows are done. Raises ``CacheMismatchError`` when this session's teacher,
    card, or libraries differ from those that started the cache: one cache must not mix
    two sets of kernels.
    """
    import torch  # noqa: PLC0415

    meta = read_meta(out)
    recorded = asdict(meta)
    for name, value in _session(facts).items():
        if recorded[name] != value:
            raise CacheMismatchError(
                f"{out / META_FILE} records {name}={recorded[name]!r}, but this session has "
                f"{value!r}; resume on the setup that started the cache, or delete it"
            )
    input_ids = np.load(out / INPUT_IDS_FILE, mmap_mode="r")
    topk_ids = np.load(out / TOPK_IDS_FILE, mmap_mode="r+")
    topk_probs = np.load(out / TOPK_PROBS_FILE, mmap_mode="r+")
    rest = np.load(out / REST_FILE, mmap_mode="r+")
    limit = meta.n_rows if stop_after is None else min(stop_after, meta.n_rows)
    session_start, earlier = time.monotonic(), meta.wall_seconds
    for first in range(meta.rows_done, limit, meta.batch_rows):
        last = min(first + meta.batch_rows, meta.n_rows)
        ids = torch.from_numpy(input_ids[first:last].astype(np.int64)).to(device)
        with torch.inference_mode():
            logits = forward(ids)[:, : meta.seq_len - 1]
            for offset, row in enumerate(range(first, last)):
                row_ids, row_probs, row_rest = teacher_targets(logits[offset], meta.top_k, row=row)
                topk_ids[row] = row_ids.cpu().numpy()
                topk_probs[row] = row_probs.cpu().numpy()
                rest[row] = row_rest.cpu().numpy()
        if last % meta.flush_rows == 0 or last >= limit:
            for target in (topk_ids, topk_probs, rest):
                target.flush()
            meta = replace(
                meta, rows_done=last, wall_seconds=earlier + time.monotonic() - session_start
            )
            _write_meta(out, meta)
    return meta


def finish_cache(out: Path) -> CacheMeta:
    """Summarise ``rest.npy`` and mark the cache complete; called once fidelity has passed."""
    meta = read_meta(out)
    if meta.rows_done != meta.n_rows:
        raise ValueError(f"cache {out} holds {meta.rows_done} of {meta.n_rows} rows; fill it first")
    rest = np.load(out / REST_FILE, mmap_mode="r")
    summary = {
        "mean": float(rest.mean(dtype=np.float64)),
        **{name: float(np.percentile(rest, q)) for name, q in _REST_PERCENTILES.items()},
        "max": float(rest.max()),
    }
    meta = replace(meta, complete=True, rest_summary=summary)
    _write_meta(out, meta)
    return meta


def _session(facts: TeacherFacts) -> dict[str, Any]:
    """The ``cache.json`` fields that pin how the targets were computed, for this session."""
    return {
        "teacher_commit": facts.commit,
        "vocab_size": facts.vocab_size,
        "eos_token_id": facts.eos_token_id,
        "tokenizer_fingerprint": facts.tokenizer_fingerprint,
        "teacher_dtype": facts.dtype,
        "device": facts.device,
        "deterministic": facts.deterministic,
        "cublas_workspace_config": facts.cublas_workspace_config,
        "versions": library_versions(),
    }
