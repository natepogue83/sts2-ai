"""Bounded ordered execution across independent native headless processes."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from time import monotonic
from typing import Callable, Generic, Sequence, TypeVar, cast


T = TypeVar("T")


@dataclass(frozen=True)
class HeadlessBatchJob(Generic[T]):
    """One episode or environment operation with a stable diagnostic key."""

    key: str
    run: Callable[[], T]


@dataclass(frozen=True)
class HeadlessBatchOutcome(Generic[T]):
    """The isolated outcome of one job, retained in input order."""

    index: int
    key: str
    value: T | None
    error: Exception | None
    elapsed_seconds: float

    @property
    def succeeded(self) -> bool:
        return self.error is None


@dataclass(frozen=True)
class HeadlessBatchReport(Generic[T]):
    """An ordered batch report that keeps failures local to their jobs."""

    outcomes: tuple[HeadlessBatchOutcome[T], ...]
    max_workers: int
    elapsed_seconds: float

    @property
    def failures(self) -> tuple[HeadlessBatchOutcome[T], ...]:
        return tuple(outcome for outcome in self.outcomes if not outcome.succeeded)

    def values(self) -> tuple[T, ...]:
        """Return ordered values, or raise after every submitted job has settled."""

        if self.failures:
            raise HeadlessBatchError(self)
        return tuple(cast(T, outcome.value) for outcome in self.outcomes)


class HeadlessBatchError(RuntimeError):
    """One or more jobs failed; successful sibling outcomes remain available."""

    def __init__(self, report: HeadlessBatchReport[object]):
        self.report = report
        summary = ", ".join(
            f"{outcome.key}: {type(outcome.error).__name__}: {outcome.error}"
            for outcome in report.failures
        )
        super().__init__(f"{len(report.failures)} headless batch job(s) failed: {summary}")


class HeadlessBatchManager:
    """Coordinate blocking calls to already-isolated native simulator processes.

    Each rollout still creates and owns its existing ``Sts2Env``/``sts2-cli``
    process. Threads only overlap blocking protocol calls, avoiding a redundant
    layer of Python multiprocessing and its Windows spawn/pickling constraints.
    """

    def __init__(self, max_workers: int = 1):
        if max_workers <= 0:
            raise ValueError("max_workers must be positive")
        self.max_workers = max_workers

    def run(self, jobs: Sequence[HeadlessBatchJob[T]]) -> HeadlessBatchReport[T]:
        keys = [job.key for job in jobs]
        if any(not key for key in keys):
            raise ValueError("Every batch job must have a non-empty key")
        if len(set(keys)) != len(keys):
            raise ValueError("Batch job keys must be unique")
        if not jobs:
            return HeadlessBatchReport((), self.max_workers, 0.0)

        started = monotonic()
        ordered: list[HeadlessBatchOutcome[T] | None] = [None] * len(jobs)
        with ThreadPoolExecutor(
            max_workers=min(self.max_workers, len(jobs)),
            thread_name_prefix="stsai-headless",
        ) as executor:
            futures = {
                executor.submit(self._invoke, index, job): index for index, job in enumerate(jobs)
            }
            for future in as_completed(futures):
                outcome = future.result()
                ordered[outcome.index] = outcome

        return HeadlessBatchReport(
            tuple(cast(HeadlessBatchOutcome[T], outcome) for outcome in ordered),
            self.max_workers,
            monotonic() - started,
        )

    @staticmethod
    def _invoke(index: int, job: HeadlessBatchJob[T]) -> HeadlessBatchOutcome[T]:
        started = monotonic()
        try:
            value = job.run()
        except Exception as exc:
            return HeadlessBatchOutcome(index, job.key, None, exc, monotonic() - started)
        return HeadlessBatchOutcome(index, job.key, value, None, monotonic() - started)
