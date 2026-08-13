"""The clock seam and the generation driver: where TTFT and ITL are actually timed.

Both clocks share one trial structure, so the clock is the only thing that changes on the
GPU and the rig stays CPU-testable. The decode loop keeps every tensor on device, since a
single host sync mid-loop would serialise the queue and inflate the fast decode steps.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from typing import Any, Protocol

from frontier.latency.stats import MS_PER_S, TrialTiming

MIN_SPANS = 2


class TrialClock(Protocol):
    """Marks instants during a trial and resolves the inter-mark spans in ms."""

    def mark(self) -> None: ...

    def resolve(self) -> list[float]: ...


class PerfCounterClock:
    """Wall-clock timing for the CPU path, where ops are synchronous and a timestamp is exact."""

    def __init__(self) -> None:
        self._marks: list[float] = []

    def mark(self) -> None:
        self._marks.append(time.perf_counter())

    def resolve(self) -> list[float]:
        marks = self._marks
        return [(marks[i + 1] - marks[i]) * MS_PER_S for i in range(len(marks) - 1)]


class CudaEventClock:
    """CUDA-event timing, the pod-only path.

    ``mark`` records an event without synchronising, so the decode loop never blocks;
    ``resolve`` synchronises exactly once and then reads each consecutive event pair.
    """

    def __init__(self, torch_module: Any) -> None:
        self._torch = torch_module
        self._events: list[Any] = []

    def mark(self) -> None:
        event = self._torch.cuda.Event(enable_timing=True)
        event.record()
        self._events.append(event)

    def resolve(self) -> list[float]:
        self._torch.cuda.synchronize()
        events = self._events
        return [float(events[i].elapsed_time(events[i + 1])) for i in range(len(events) - 1)]


class GenerationDriver(Protocol):
    """Drives one timed autoregressive trial, marking ``clock`` per the TTFT/ITL order."""

    def run_trial(
        self, clock: TrialClock, *, batch_size: int, context_len: int, decode_len: int
    ) -> None: ...


class HFGenerationDriver:
    """Drives timed decode over an already-loaded HF model and tokenizer.

    Built from ``HFLogitProvider.loaded_model`` so the weights load once for both scoring
    and timing. Timing is shape-driven: the prefill is a ``(batch_size, context_len)``
    block of one mid-vocab filler token. The first pair of marks bounds TTFT (before
    prefill, after the first token), then one mark per decode step.
    """

    def __init__(self, model: Any, tokenizer: Any, device: str) -> None:
        self._model = model
        self._device = device
        self._filler_id = _mid_vocab_token(int(model.config.vocab_size), tokenizer.all_special_ids)

    def run_trial(
        self, clock: TrialClock, *, batch_size: int, context_len: int, decode_len: int
    ) -> None:
        import torch  # noqa: PLC0415
        import transformers  # noqa: PLC0415

        model = self._model
        input_ids = torch.full(
            (batch_size, context_len), self._filler_id, dtype=torch.long, device=self._device
        )
        attention_mask = torch.ones(
            (batch_size, context_len), dtype=torch.long, device=self._device
        )
        cache = transformers.DynamicCache(config=model.config)
        with torch.inference_mode():
            clock.mark()
            # logits_to_keep=1: full prefill logits run ~10GB at batch 16 / context 2048,
            # an OOM on a 16GB card, and only the last position drives the next token.
            output = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=cache,
                use_cache=True,
                logits_to_keep=1,
            )
            next_ids = output.logits[:, -1:, :].argmax(-1)
            clock.mark()
            for _ in range(decode_len - 1):
                attention_mask = torch.cat(
                    [attention_mask, attention_mask.new_ones((batch_size, 1))], dim=-1
                )
                output = model(
                    input_ids=next_ids,
                    attention_mask=attention_mask,
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                )
                next_ids = output.logits[:, -1:, :].argmax(-1)
                clock.mark()


def collect_trials(
    driver: GenerationDriver,
    clock_factory: Callable[[], TrialClock],
    *,
    batch_size: int,
    context_len: int,
    decode_len: int,
    n_total: int,
) -> list[TrialTiming]:
    """Run ``n_total`` trials (warmup + kept), one ``TrialTiming`` each.

    Each trial gets a fresh clock, whose ``spans[0]`` is the TTFT and ``spans[1:]`` the
    per-token ITL samples, so ``decode_len`` must be at least 2 for one ITL sample.
    """
    trials: list[TrialTiming] = []
    for _ in range(n_total):
        clock = clock_factory()
        driver.run_trial(
            clock, batch_size=batch_size, context_len=context_len, decode_len=decode_len
        )
        spans = clock.resolve()
        if len(spans) < MIN_SPANS:
            raise ValueError(
                f"trial produced {len(spans)} spans; need >= {MIN_SPANS} "
                f"(decode_len={decode_len} must be >= 2 for one ITL sample)"
            )
        trials.append(TrialTiming(ttft_ms=spans[0], itl_ms=tuple(spans[1:])))
    return trials


def _mid_vocab_token(vocab_size: int, special_ids: Sequence[int]) -> int:
    special = set(special_ids)
    token = vocab_size // 2
    while token in special and token + 1 < vocab_size:
        token += 1
    return token
