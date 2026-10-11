"""Asynchronous estimation service (single worker thread + queue)."""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from typing import Optional

import numpy as np

from .domain import EstimateOutcome
from .ports import EstimatorPort, EstimateError, ModelUnavailableError


@dataclass
class EstimateResult:
    """A marshalled worker result delivered to the main thread by ``poll``."""

    token: int
    sample: str
    epoch: int
    outcome: Optional[EstimateOutcome] = None
    error: Optional[str] = None
    error_kind: Optional[str] = None  # "model" | "estimate"


@dataclass
class _Job:
    token: int
    sample: str
    image: np.ndarray
    epoch: int


class EstimationService:
    """Runs the estimator on one worker thread.

    Estimates are serialised because :data:`geltilt.ladders.P` is a module-global
    mutated during a call. At most one request is pending; a newer request
    supersedes an older one by token. Results are delivered through a queue and
    consumed by the main thread via :meth:`poll`.
    """

    def __init__(self, estimator: EstimatorPort) -> None:
        self._estimator = estimator
        self._queue: "queue.Queue[EstimateResult]" = queue.Queue()
        self._lock = threading.Lock()
        self._token = 0
        self._latest = 0
        self._pending: Optional[_Job] = None
        self._thread: Optional[threading.Thread] = None

    # -- lifecycle ---------------------------------------------------------- #
    def start(self, sample: str, image: np.ndarray, epoch: int) -> int:
        """Submit a request; returns the new token (supersedes earlier ones)."""
        with self._lock:
            self._token += 1
            token = self._token
            self._latest = token
            self._pending = _Job(token=token, sample=sample, image=image, epoch=epoch)
            need_thread = self._thread is None or not self._thread.is_alive()
            if need_thread:
                self._thread = threading.Thread(target=self._worker, daemon=True)
                thread = self._thread
            else:
                thread = None
        if thread is not None:
            thread.start()
        return token

    def cancel(self) -> None:
        """Drop any pending request and mark the in-flight result stale."""
        with self._lock:
            self._latest = self._token + 1
            self._token = self._latest
            self._pending = None

    @property
    def busy(self) -> bool:
        with self._lock:
            if self._pending is not None:
                return True
            return self._thread is not None and self._thread.is_alive()

    # -- worker ------------------------------------------------------------- #
    def _worker(self) -> None:
        while True:
            with self._lock:
                job = self._pending
                self._pending = None
                if job is None:
                    self._thread = None
                    return
                latest = self._latest
            if job.token != latest:
                continue  # superseded before we started
            result = self._run_job(job)
            with self._lock:
                if job.token == self._latest:
                    self._queue.put(result)

    def _run_job(self, job: _Job) -> EstimateResult:
        try:
            outcome = self._estimator.estimate(job.image)
            return EstimateResult(job.token, job.sample, job.epoch, outcome=outcome)
        except ModelUnavailableError as exc:
            return EstimateResult(
                job.token, job.sample, job.epoch, error=str(exc), error_kind="model"
            )
        except EstimateError as exc:
            return EstimateResult(
                job.token, job.sample, job.epoch, error=str(exc), error_kind="estimate"
            )
        except Exception as exc:  # noqa: BLE001 - isolate any estimator failure
            return EstimateResult(
                job.token, job.sample, job.epoch, error=str(exc), error_kind="estimate"
            )

    # -- main-thread polling ------------------------------------------------ #
    def poll(self) -> Optional[EstimateResult]:
        try:
            result = self._queue.get_nowait()
        except queue.Empty:
            return None
        with self._lock:
            if result.token != self._latest:
                return None  # stale: superseded or cancelled
        return result