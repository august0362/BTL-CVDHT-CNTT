"""Session-owned background jobs. Workers never access Streamlit or widget state."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from threading import RLock
from time import monotonic
from typing import Any, Callable, MutableMapping
from uuid import uuid4


_POOL = ThreadPoolExecutor(max_workers=4, thread_name_prefix="ma3d-job")


@dataclass(frozen=True)
class JobSnapshot:
    id: str
    key: Any
    state: str
    revision: int
    message: str
    progress: Any
    result: Any
    error: str | None

    @property
    def active(self) -> bool:
        return self.state in ("queued", "running")


class Job:
    def __init__(self, key: Any):
        self.id, self.key = uuid4().hex, key
        self._lock = RLock()
        self._state, self._revision = "queued", 0
        self._message = "Đang chuẩn bị…"
        self._progress = self._result = self._error = None
        self.started = monotonic()
        self.future = None

    def publish(self, message: str | None = None, progress: Any = None) -> None:
        """Publish a new, completed snapshot; callers must not mutate it afterward."""
        with self._lock:
            if message is not None:
                self._message = message
            if progress is not None:
                self._progress = progress
            self._revision += 1

    def snapshot(self) -> JobSnapshot:
        with self._lock:
            return JobSnapshot(self.id, self.key, self._state, self._revision,
                               self._message, self._progress, self._result, self._error)

    def _run(self, work: Callable[[Callable], Any]) -> None:
        with self._lock:
            self._state = "running"
            self._revision += 1
        try:
            result = work(self.publish)
        except Exception as exc:
            with self._lock:
                self._state, self._error = "error", f"{type(exc).__name__}: {exc}"
                self._revision += 1
        else:
            with self._lock:
                self._state, self._result = "done", result
                self._message = "Đã hoàn tất"
                self._revision += 1


class SessionJobs:
    """Independent channels; replacing a request cannot publish an old result over it."""
    def __init__(self):
        self._lock = RLock()
        self._jobs: dict[str, Job] = {}

    def submit(self, channel: str, key: Any, work: Callable[[Callable], Any], *, force=False) -> Job:
        with self._lock:
            previous = self._jobs.get(channel)
            if previous is not None and previous.key == key:
                snap = previous.snapshot()
                if snap.active or not force:
                    return previous
            job = Job(key)
            self._jobs[channel] = job
            # A queued obsolete job can be cancelled. Running work owns its old
            # Job and can finish without changing the replacement channel.
            if previous is not None and previous.future is not None:
                previous.future.cancel()
            job.future = _POOL.submit(job._run, work)
            return job

    def get(self, channel: str) -> Job | None:
        with self._lock:
            return self._jobs.get(channel)

    def discard(self, channel: str) -> None:
        with self._lock:
            old = self._jobs.pop(channel, None)
            if old is not None and old.future is not None:
                old.future.cancel()


def session_jobs(state: MutableMapping) -> SessionJobs:
    """Call only on the UI thread; the returned object itself is thread safe."""
    if "_background_jobs" not in state:
        state["_background_jobs"] = SessionJobs()
    return state["_background_jobs"]
