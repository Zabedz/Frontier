"""Build or resume a distill variant's teacher cache, check its fidelity, and score the
untrained student on the held-out rows."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from frontier.backends.hf import resolve_dtype
from frontier.distill.cache import (
    CACHE_FORMAT,
    FLUSH_EVERY_ROWS,
    META_FILE,
    TEACHER_BATCH_ROWS,
    CacheMeta,
    CacheMismatchError,
    CacheRequest,
    TeacherCache,
    TeacherFacts,
    device_name,
    fill_targets,
    finish_cache,
    read_meta,
    start_cache,
    tokenizer_fingerprint,
)
from frontier.distill.fidelity import check_fidelity
from frontier.distill.heldout import (
    BASE_HELDOUT_FILE,
    heldout_report,
    read_heldout,
    write_heldout,
)
from frontier.distill.paths import cache_dir
from frontier.distill.spec import (
    SERVED_WEIGHT_DTYPE,
    ModelRef,
    require_distill_spec,
    student_model,
    teacher_model,
    token_budget,
)
from frontier.schema import VariantConfig
from frontier.training.corpus import TextStream, pack_tokens, stream_corpus

HELDOUT_ROWS = 64
SMOKE_HELDOUT_ROWS = 4  # held-out rows for a capped build (smoke, timing)
# cuBLAS picks deterministic kernels only with this workspace, read when CUDA initialises.
CUBLAS_WORKSPACE_ENV = "CUBLAS_WORKSPACE_CONFIG"
DETERMINISTIC_CUBLAS_WORKSPACE = ":4096:8"


def cache_request(variant: VariantConfig, *, max_tokens: int | None = None) -> CacheRequest:
    """What ``variant`` needs from the cache at the budget ``max_tokens`` leaves it.

    Raises ``ValueError`` for a budget under one row.
    """
    spec = require_distill_spec(variant)
    teacher = teacher_model(variant)
    budget = token_budget(spec, max_tokens)
    train_rows = budget // spec.seq_len
    if train_rows == 0:
        raise ValueError(f"a {budget}-token budget is under one {spec.seq_len}-token row")
    capped = budget < spec.train_tokens
    return CacheRequest(
        teacher_id=teacher.model_id,
        teacher_revision=teacher.revision,
        corpus=spec.corpus,
        seq_len=spec.seq_len,
        top_k=spec.top_k,
        train_rows=train_rows,
        heldout_rows=SMOKE_HELDOUT_ROWS if capped else HELDOUT_ROWS,
        budget_tokens=budget,
        capped=capped,
    )


def ensure_cache(
    variant: VariantConfig,
    *,
    checkpoints_root: Path,
    device: str,
    max_tokens: int | None = None,
    texts: TextStream | None = None,
) -> TeacherCache:
    """The variant's teacher cache, built or resumed to completion and fidelity-checked.

    A complete cache returns without loading a model. The build and both fidelity checks run
    under deterministic kernels, and the untrained student is scored on the held-out rows
    once the teacher has gone. ``max_tokens`` caps the training rows (smoke, timing);
    ``texts`` replaces the streamed corpus, for tests.

    Raises:
        CacheMismatchError: the cache on disk was built for another teacher, corpus, row
            length, k, or format, or holds fewer training rows than this run needs.
        CacheFidelityError: the cache disagrees with a recompute of the teacher.
    """
    import torch  # noqa: PLC0415

    request = cache_request(variant, max_tokens=max_tokens)
    path = cache_dir(variant, root=checkpoints_root)
    if (path / META_FILE).is_file():
        meta = read_meta(path)
        _check_request(path, meta, request)
        if meta.complete:
            cache = TeacherCache(path)
            if not _base_scored(variant, cache):
                _score_base_student(variant, cache, device=device)
            return cache
    teacher = teacher_model(variant)
    with _deterministic_kernels():
        tokenizer, config = _tokenizer_and_config(teacher)
        facts = _teacher_facts(tokenizer, config, device=device)
        if not (path / META_FILE).is_file():
            _check_student_matches(variant, facts, top_k=request.top_k)
            _start(path, request, facts, tokenizer=tokenizer, texts=texts)
        _run_teacher(path, teacher, facts, device=device)
    if device != "cpu":
        torch.cuda.empty_cache()
    finish_cache(path)
    cache = TeacherCache(path)
    _score_base_student(variant, cache, device=device)
    return cache


def served_dtype(device: str) -> str:
    """The torch dtype name the hf backend serves an fp16 variant in on ``device``."""
    return resolve_dtype(device, SERVED_WEIGHT_DTYPE)


def deterministic_cublas() -> None:
    """Ask cuBLAS for deterministic kernels; must run before CUDA initialises."""
    os.environ.setdefault(CUBLAS_WORKSPACE_ENV, DETERMINISTIC_CUBLAS_WORKSPACE)


def _check_request(path: Path, meta: CacheMeta, request: CacheRequest) -> None:
    wanted = {
        "format": CACHE_FORMAT,
        "teacher_id": request.teacher_id,
        "teacher_revision": request.teacher_revision,
        "corpus": request.corpus,
        "seq_len": request.seq_len,
        "top_k": request.top_k,
    }
    for name, value in wanted.items():
        recorded = getattr(meta, name)
        if recorded != value:
            raise CacheMismatchError(
                f"teacher cache {path} was built with {name}={recorded!r}; this run needs {value!r}"
            )
    available = meta.n_rows - meta.heldout_rows
    if available < request.train_rows:
        raise CacheMismatchError(
            f"teacher cache {path} holds {available} training rows and this run needs "
            f"{request.train_rows}; a capped cache cannot serve a larger run, so delete it or "
            f"pass another --checkpoints"
        )


@contextmanager
def _deterministic_kernels() -> Iterator[None]:
    """Deterministic algorithms for the build and its checks, restored after.

    An op with no deterministic kernel warns; the fidelity check measures what that costs.
    """
    import torch  # noqa: PLC0415

    enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.use_deterministic_algorithms(True, warn_only=True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(enabled, warn_only=warn_only)


def _tokenizer_and_config(model: ModelRef) -> tuple[Any, Any]:
    import transformers  # noqa: PLC0415

    tokenizer = transformers.AutoTokenizer.from_pretrained(model.model_id, revision=model.revision)
    config = transformers.AutoConfig.from_pretrained(model.model_id, revision=model.revision)
    return tokenizer, config


def _teacher_facts(tokenizer: Any, config: Any, *, device: str) -> TeacherFacts:
    import torch  # noqa: PLC0415

    if tokenizer.eos_token_id is None:
        raise ValueError("the teacher tokenizer has no eos_token_id to separate documents")
    return TeacherFacts(
        vocab_size=int(config.vocab_size),
        eos_token_id=int(tokenizer.eos_token_id),
        tokenizer_fingerprint=tokenizer_fingerprint(tokenizer),
        commit=getattr(config, "_commit_hash", None),
        dtype=served_dtype(device),
        device=device_name(device),
        deterministic=torch.are_deterministic_algorithms_enabled(),
        cublas_workspace_config=os.environ.get(CUBLAS_WORKSPACE_ENV),
    )


def _check_student_matches(variant: VariantConfig, teacher: TeacherFacts, *, top_k: int) -> None:
    """The student must read the teacher's token ids: one tokenizer, one vocabulary."""
    student = student_model(variant)
    tokenizer, config = _tokenizer_and_config(student)
    found = {
        "tokenizer": (tokenizer_fingerprint(tokenizer), teacher.tokenizer_fingerprint),
        "eos_token_id": (tokenizer.eos_token_id, teacher.eos_token_id),
        "vocab_size": (int(config.vocab_size), teacher.vocab_size),
    }
    for name, (student_value, teacher_value) in found.items():
        if student_value != teacher_value:
            raise ValueError(
                f"student {student.model_id} differs from teacher {variant.model.model_id} in "
                f"{name} ({student_value!r} against {teacher_value!r}); distillation through the "
                f"cache needs one tokenizer"
            )
    if top_k >= teacher.vocab_size:
        raise ValueError(
            f"top_k {top_k} leaves no rest bucket in a {teacher.vocab_size}-entry vocabulary"
        )


def _start(
    path: Path,
    request: CacheRequest,
    facts: TeacherFacts,
    *,
    tokenizer: Any,
    texts: TextStream | None,
) -> None:
    """Pack the training and held-out rows with the teacher tokenizer and start the cache."""
    stream = (texts or stream_corpus)(request.corpus)
    rows = pack_tokens(
        stream,
        tokenizer,
        seq_len=request.seq_len,
        max_tokens=(request.train_rows + request.heldout_rows) * request.seq_len,
    )
    # An abandoned streamed parquet read blocks pyarrow's thread-pool shutdown, so the
    # process would hang at exit; closing the generator ends the read.
    close = getattr(stream, "close", None)
    if close is not None:
        close()
    start_cache(
        path, rows, request, facts, batch_rows=TEACHER_BATCH_ROWS, flush_rows=FLUSH_EVERY_ROWS
    )


def _run_teacher(path: Path, teacher: ModelRef, facts: TeacherFacts, *, device: str) -> None:
    """Fill the targets past the watermark and check them; the teacher is freed on return.

    Until the first flush has passed, the first batch is checked on its own right after it,
    so a recompute mismatch stops a build, or its relaunch, within minutes.
    """
    import torch  # noqa: PLC0415
    import transformers  # noqa: PLC0415

    model = transformers.AutoModelForCausalLM.from_pretrained(
        teacher.model_id, revision=teacher.revision, dtype=getattr(torch, served_dtype(device))
    )
    model.config.use_cache = False
    model.to(device).eval()

    def forward(ids: Any) -> Any:
        return model(input_ids=ids).logits

    meta = read_meta(path)
    if meta.rows_done <= meta.flush_rows:
        fill_targets(path, forward, facts=facts, device=device, stop_after=meta.flush_rows)
        check_fidelity(path, forward, device=device, batches=1)
    fill_targets(path, forward, facts=facts, device=device)
    check_fidelity(path, forward, device=device)


def _base_scored(variant: VariantConfig, cache: TeacherCache) -> bool:
    """Whether ``heldout-base.json`` holds this variant's student.

    Raises ``ValueError`` when it holds another student: two students on one cache need
    separate checkpoint roots.
    """
    path = cache.path / BASE_HELDOUT_FILE
    if not path.is_file():
        return False
    scored = read_heldout(path).student_id
    student = student_model(variant).model_id
    if scored != student:
        raise ValueError(
            f"{path} scores the untrained {scored}, not {student}; give a second student on "
            f"this cache its own --checkpoints"
        )
    return True


def _score_base_student(variant: VariantConfig, cache: TeacherCache, *, device: str) -> None:
    """``heldout-base.json``: the untrained student, served as the trained ones are."""
    import torch  # noqa: PLC0415
    import transformers  # noqa: PLC0415

    student = student_model(variant)
    model = transformers.AutoModelForCausalLM.from_pretrained(
        student.model_id, revision=student.revision, dtype=getattr(torch, served_dtype(device))
    ).to(device)
    report = heldout_report(model, cache, student_id=student.model_id, device=device)
    write_heldout(cache.path / BASE_HELDOUT_FILE, report)
