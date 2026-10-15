"""Background task isolation, safe publication and navigation without model inference."""
from pathlib import Path
import sys
from threading import Event, Lock

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT / "backend" / "web"))
sys.path.insert(0, str(ROOT / "frontend"))

from background_jobs import SessionJobs, session_jobs


def test_channels_run_independently_and_repeated_reruns_do_not_duplicate_work():
    jobs = SessionJobs()
    main_started, second_started, finish_main = Event(), Event(), Event()
    calls = []

    def main(report):
        calls.append("main")
        report(message="Working on main")
        main_started.set()
        assert finish_main.wait(5)
        return "main result"

    def second_channel(report):
        calls.append("second")
        second_started.set()
        return "second result"

    first = jobs.submit("main", "a", main)
    try:
        assert main_started.wait(2)
        assert jobs.submit("main", "a", main) is first
        second = jobs.submit("secondary", "b", second_channel)
        assert second_started.wait(2)
        second.future.result(2)
        assert first.snapshot().active
        assert second.snapshot().result == "second result"
    finally:
        finish_main.set()
        first.future.result(3)
    assert first.snapshot().result == "main result"
    assert calls.count("main") == calls.count("second") == 1


def test_a_late_obsolete_result_cannot_overwrite_a_new_request():
    jobs = SessionJobs()
    started, release = Event(), Event()

    def old_work(report):
        started.set()
        assert release.wait(5)
        return "old"

    old = jobs.submit("main", "old image", old_work)
    try:
        assert started.wait(2)
        new = jobs.submit("main", "new image", lambda report: "new")
        new.future.result(2)
        release.set()
        old.future.result(2)
        assert jobs.get("main") is new
        assert jobs.get("main").snapshot().result == "new"
    finally:
        release.set()


def test_errors_are_isolated_and_retry_requires_a_new_request_or_explicit_force():
    jobs = SessionJobs()
    count = []

    def broken(report):
        count.append(1)
        raise ValueError("invalid input")

    job = jobs.submit("main", "x", broken)
    job.future.result(2)
    assert job.snapshot().state == "error"
    assert job.snapshot().error == "ValueError: invalid input"
    assert jobs.submit("main", "x", broken) is job
    assert len(count) == 1
    retry = jobs.submit("main", "x", lambda report: "ok", force=True)
    retry.future.result(2)
    assert retry.snapshot().result == "ok"


def test_sessions_and_published_snapshots_are_separate():
    a, b = {}, {}
    assert session_jobs(a) is session_jobs(a)
    assert session_jobs(a) is not session_jobs(b)
    ready, finish = Event(), Event()

    def work(report):
        report(progress={"phase": "first"})
        ready.set()
        assert finish.wait(4)
        report(progress={"phase": "second"})
        return "ok"

    job = session_jobs(a).submit("secondary", "x", work)
    try:
        assert ready.wait(2)
        before = job.snapshot()
        assert before.progress == {"phase": "first"}
        assert session_jobs(b).get("secondary") is None
    finally:
        finish.set()
        job.future.result(2)
    assert before.progress == {"phase": "first"}
    assert job.snapshot().progress == {"phase": "second"}
    assert job.snapshot().revision > before.revision


def test_shared_model_stages_do_not_overlap_and_allow_reentrant_calls():
    from web.runtime_sync import model_stage, MODEL_RUNTIME_LOCK
    from concurrent.futures import ThreadPoolExecutor
    entered, release = Event(), Event()
    order, guard = [], Lock()

    @model_stage
    def first():
        with MODEL_RUNTIME_LOCK:
            with guard:
                order.append("first start")
            entered.set()
            assert release.wait(4)
            with guard:
                order.append("first end")

    @model_stage
    def second():
        with guard:
            order.append("second")

    with ThreadPoolExecutor(max_workers=2) as pool:
        a = pool.submit(first)
        try:
            assert entered.wait(2)
            b = pool.submit(second)
            assert "second" not in order
        finally:
            release.set()
        a.result(2)
        b.result(2)
    assert order == ["first start", "first end", "second"]
