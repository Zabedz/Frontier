"""A student's fit to the teacher and the corpus on the held-out rows no trainer reads: did
the KD move the student toward the teacher, whatever MMLU shows."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from frontier.distill.cache import TeacherCache, write_json
from frontier.distill.losses import LOSS_CHUNK_POSITIONS, soft_topk_kl, target_tensors, token_nll

HELDOUT_FILE = "heldout.json"  # beside each trained student
BASE_HELDOUT_FILE = "heldout-base.json"  # in the cache directory, for the untrained student
HELDOUT_BATCH_ROWS = 4


@dataclass(frozen=True, slots=True)
class HeldoutReport:
    """Means over every held-out position."""

    student_id: str  # the hub student the scored weights start from
    rows: int
    positions: int
    soft_kl: float  # the soft-label loss: KL from the teacher's top-k-plus-rest target
    hard_nll: float  # cross-entropy against the teacher's top-1
    corpus_nll: float  # cross-entropy against the corpus's next token
    top1_agreement: float  # fraction of positions where the argmax is the teacher's top-1


def heldout_report(
    model: Any, cache: TeacherCache, *, student_id: str, device: str
) -> HeldoutReport:
    """Score ``model`` on the held-out rows, in eval mode and the dtype it was loaded in."""
    import torch  # noqa: PLC0415

    decoder, head = model.get_decoder(), model.get_output_embeddings()
    rows = np.arange(cache.heldout.start, cache.heldout.stop, dtype=np.int64)
    soft = hard = corpus = agree = 0.0
    positions = 0
    model.eval()
    with torch.inference_mode():
        for first in range(0, rows.shape[0], HELDOUT_BATCH_ROWS):
            batch = cache.batch(rows[first : first + HELDOUT_BATCH_ROWS], "soft_label_topk")
            hidden = decoder(input_ids=torch.from_numpy(batch.input_ids).to(device))
            flat = hidden.last_hidden_state[:, :-1].reshape(-1, hidden.last_hidden_state.shape[-1])
            n = int(flat.shape[0])
            ids, probs, rest = (
                t.reshape(n, *t.shape[2:])
                for t in target_tensors(batch, "soft_label_topk", device=device)
            )
            (following,) = (
                t.reshape(n) for t in target_tensors(batch, "corpus_label", device=device)
            )
            for start in range(0, n, LOSS_CHUNK_POSITIONS):
                span = slice(start, start + LOSS_CHUNK_POSITIONS)
                logits = head(flat[span]).float()
                top1 = ids[span, 0]
                soft += float(
                    soft_topk_kl(logits, ids[span], probs[span], rest[span]).double().sum()
                )
                hard += float(token_nll(logits, top1).double().sum())
                corpus += float(token_nll(logits, following[span]).double().sum())
                agree += float((logits.argmax(dim=-1) == top1).sum())
            positions += n
    return HeldoutReport(
        student_id=student_id,
        rows=int(rows.shape[0]),
        positions=positions,
        soft_kl=soft / positions,
        hard_nll=hard / positions,
        corpus_nll=corpus / positions,
        top1_agreement=agree / positions,
    )


def write_heldout(path: Path, report: HeldoutReport) -> None:
    write_json(path, asdict(report))


def read_heldout(path: Path) -> HeldoutReport:
    return HeldoutReport(**json.loads(path.read_text(encoding="utf-8")))
