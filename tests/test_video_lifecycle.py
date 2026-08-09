"""Coverage for the video lifecycle's failure-mode + cancellation paths.

Three focal areas:
  1. ``ComfyUIClient.poll_completion`` detects a dropped prompt via /queue.
  2. ``TaskScheduler.cancel`` actually cancels a running task by name.
  3. ``DELETE /v1/videos/{id}`` releases a stuck slot end-to-end.
"""

from __future__ import annotations

import asyncio
import json
import textwrap
from collections.abc import Iterator
from pathlib import Path

import httpx
import pytest
import respx
from fastapi.testclient import TestClient

from openai_api_bridge.backends.comfyui import client as comfy_client_module
from openai_api_bridge.backends.comfyui.client import ComfyUIClient
from openai_api_bridge.config import reset_caches_for_tests
from openai_api_bridge.errors import UpstreamError
from openai_api_bridge.infra.jobstore import JobStore
from openai_api_bridge.infra.tasks import TaskScheduler

# --- 1. poll_completion detects dropped prompts -------------------------


@respx.mock
@pytest.mark.asyncio
async def test_poll_completion_fails_fast_when_comfyui_drops_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Tighten both the recheck interval and the miss threshold so the test
    # runs in well under a second instead of waiting the production 30s x 3.
    monkeypatch.setattr(comfy_client_module, "QUEUE_RECHECK_INTERVAL", 0.05)
    monkeypatch.setattr(comfy_client_module, "QUEUE_MISS_THRESHOLD", 2)

    # /history always 200 with empty body — the prompt isn't there.
    respx.get("http://comfy/history/dropped-id").mock(return_value=httpx.Response(200, json={}))
    # /queue returns no record of our prompt for every check.
    respx.get("http://comfy/queue").mock(
        return_value=httpx.Response(
            200,
            json={
                "queue_running": [],
                "queue_pending": [[0, "some-other-prompt", {}, {}]],
            },
        )
    )

    client = ComfyUIClient(base_url="http://comfy", poll_interval_seconds=0.01)
    try:
        with pytest.raises(UpstreamError, match="dropped prompt dropped-id"):
            # Generous outer timeout so a slow CI doesn't false-positive on the
            # GenerationTimeout path; the queue check should fire well before this.
            await client.poll_completion("dropped-id", timeout_seconds=10.0)
    finally:
        await client.aclose()


@respx.mock
@pytest.mark.asyncio
async def test_poll_completion_tolerates_transient_queue_miss(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One queue miss isn't enough to declare a drop — the streak resets if
    the prompt reappears. Defends against custom-node transitions where the
    /queue endpoint briefly misreports during heavy execution."""
    monkeypatch.setattr(comfy_client_module, "QUEUE_RECHECK_INTERVAL", 0.05)
    monkeypatch.setattr(comfy_client_module, "QUEUE_MISS_THRESHOLD", 3)

    # /history empty for a while, then returns the result.
    respx.get("http://comfy/history/transient-id").mock(
        side_effect=[
            httpx.Response(200, json={}),
            httpx.Response(200, json={}),
            httpx.Response(200, json={}),
            httpx.Response(200, json={}),
            httpx.Response(200, json={}),
            httpx.Response(200, json={"transient-id": {"outputs": {}}}),
        ]
    )
    # /queue alternates: present, missing, present, missing, present.
    # Never 3 misses in a row, so the streak should reset and we should NOT
    # raise UpstreamError; eventually history returns success.
    present = httpx.Response(
        200,
        json={"queue_running": [[0, "transient-id", {}, {}]], "queue_pending": []},
    )
    missing = httpx.Response(200, json={"queue_running": [], "queue_pending": []})
    respx.get("http://comfy/queue").mock(
        side_effect=[present, missing, present, missing, present, missing]
    )

    client = ComfyUIClient(base_url="http://comfy", poll_interval_seconds=0.01)
    try:
        result = await client.poll_completion("transient-id", timeout_seconds=10.0)
        assert result == {"outputs": {}}
    finally:
        await client.aclose()


@respx.mock
@pytest.mark.asyncio
async def test_poll_completion_keeps_polling_when_prompt_still_queued(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Same low recheck — but this time /queue confirms the prompt is alive,
    # then /history eventually returns success. Should complete normally.
    monkeypatch.setattr(comfy_client_module, "QUEUE_RECHECK_INTERVAL", 0.05)

    # Two empty history responses, then a successful one.
    respx.get("http://comfy/history/active-id").mock(
        side_effect=[
            httpx.Response(200, json={}),
            httpx.Response(200, json={}),
            httpx.Response(200, json={"active-id": {"outputs": {"1": {}}}}),
        ]
    )
    respx.get("http://comfy/queue").mock(
        return_value=httpx.Response(
            200,
            json={"queue_running": [[0, "active-id", {}, {}]], "queue_pending": []},
        )
    )

    client = ComfyUIClient(base_url="http://comfy", poll_interval_seconds=0.01)
    try:
        result = await client.poll_completion("active-id", timeout_seconds=10.0)
        assert result == {"outputs": {"1": {}}}
    finally:
        await client.aclose()


# --- 2. TaskScheduler.cancel ------------------------------------------


@pytest.mark.asyncio
async def test_scheduler_cancel_releases_semaphore() -> None:
    scheduler = TaskScheduler(max_concurrent=1)

    cancel_seen = asyncio.Event()
    started = asyncio.Event()

    async def long_task() -> None:
        started.set()
        try:
            await asyncio.sleep(60)  # would block the slot for a minute
        except asyncio.CancelledError:
            cancel_seen.set()
            raise

    task = scheduler.submit(long_task(), name="job-1")
    await started.wait()  # ensure it acquired the semaphore

    assert scheduler.cancel("job-1") is True
    await cancel_seen.wait()
    # Wait for the wrapper task to fully unwind so the permit is released.
    await asyncio.gather(task, return_exceptions=True)

    # A new job should be able to acquire immediately — no deadlock.
    second_done = asyncio.Event()

    async def quick() -> None:
        second_done.set()

    scheduler.submit(quick(), name="job-2")
    await asyncio.wait_for(second_done.wait(), timeout=1.0)


@pytest.mark.asyncio
async def test_scheduler_cancel_unknown_task_returns_false() -> None:
    scheduler = TaskScheduler(max_concurrent=1)
    assert scheduler.cancel("never-existed") is False


@pytest.mark.asyncio
async def test_scheduler_hard_timeout_releases_permit() -> None:
    scheduler = TaskScheduler(max_concurrent=1, default_task_timeout_seconds=0.1)

    async def hangs_forever() -> None:
        await asyncio.sleep(60)

    task = scheduler.submit(hangs_forever(), name="hung")
    await asyncio.gather(task, return_exceptions=True)

    # A follow-up should run immediately — the timeout released the permit.
    fired = asyncio.Event()

    async def quick() -> None:
        fired.set()

    scheduler.submit(quick(), name="ok")
    await asyncio.wait_for(fired.wait(), timeout=1.0)


# --- 3. DELETE /v1/videos/{id} -----------------------------------------


@pytest.fixture
def comfyui_workflows_dir(tmp_path: Path) -> Path:
    d = tmp_path / "workflows"
    d.mkdir()
    # A video workflow: VHS_VideoCombine triggers output_type=video auto-detection.
    (d / "tiny-t2v.json").write_text(
        json.dumps(
            {
                "1": {"class_type": "CLIPTextEncode", "inputs": {"text": "x"}},
                "2": {"class_type": "VHS_VideoCombine", "inputs": {}},
            }
        )
    )
    (d / "tiny-t2v.meta.json").write_text(json.dumps({"positive_prompt_node": "1"}))
    return d


@pytest.fixture
def client_with_video(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    comfyui_workflows_dir: Path,
) -> Iterator[TestClient]:
    config = tmp_path / "config.toml"
    config.write_text(
        textwrap.dedent(f"""
        [[providers]]
        id = "comfyui"
        backend = "comfyui"
        url = "http://127.0.0.1:8188"
        workflows_dir = "{comfyui_workflows_dir}"
    """)
    )

    monkeypatch.setenv("BRIDGE_API_KEY", "test-bridge-api-key")
    monkeypatch.setenv("BRIDGE_CONFIG_PATH", str(config))
    monkeypatch.setenv("FILES_DIR", str(tmp_path / "files"))
    monkeypatch.setenv("SQLITE_PATH", str(tmp_path / "state.db"))
    monkeypatch.setenv("LOG_LEVEL", "WARNING")
    reset_caches_for_tests()

    from openai_api_bridge.main import create_app

    app = create_app()
    with TestClient(app) as c:
        yield c
    reset_caches_for_tests()


@respx.mock
def test_delete_video_cancels_in_progress_job(client_with_video: TestClient) -> None:
    """A queued/in-progress video job should be cancellable via DELETE."""
    headers = {"Authorization": "Bearer test-bridge-api-key"}

    # ComfyUI accepts the prompt — the runner will then poll forever
    # (we never stub /history with a completion).
    respx.post("http://127.0.0.1:8188/prompt").mock(
        return_value=httpx.Response(200, json={"prompt_id": "stuck-prompt"})
    )
    respx.get("http://127.0.0.1:8188/history/stuck-prompt").mock(
        return_value=httpx.Response(200, json={})
    )
    # Queue check confirms the prompt is "running" — so the runner won't
    # bail out on its own; only DELETE should end it.
    respx.get("http://127.0.0.1:8188/queue").mock(
        return_value=httpx.Response(
            200,
            json={"queue_running": [[0, "stuck-prompt", {}, {}]], "queue_pending": []},
        )
    )

    r = client_with_video.post(
        "/v1/videos",
        headers=headers,
        data={"model": "comfyui/tiny-t2v", "prompt": "x"},
    )
    assert r.status_code == 200
    job_id = r.json()["id"]

    # Wait for the runner to actually start (status flips to in_progress).
    import time

    for _ in range(40):
        body = client_with_video.get(f"/v1/videos/{job_id}", headers=headers).json()
        if body["status"] == "in_progress":
            break
        time.sleep(0.05)
    else:
        raise AssertionError("Runner never reached in_progress")

    # Cancel.
    r = client_with_video.delete(f"/v1/videos/{job_id}", headers=headers)
    assert r.status_code == 200
    assert r.json()["status"] == "failed"
    assert "cancel" in (r.json()["error"] or {}).get("message", "").lower()

    # Subsequent DELETE on a terminal job is a no-op (still 200).
    r2 = client_with_video.delete(f"/v1/videos/{job_id}", headers=headers)
    assert r2.status_code == 200
    assert r2.json()["status"] == "failed"


def test_delete_video_unknown_returns_404(client_with_video: TestClient) -> None:
    headers = {"Authorization": "Bearer test-bridge-api-key"}
    r = client_with_video.delete("/v1/videos/does-not-exist", headers=headers)
    assert r.status_code == 404
    assert r.json()["error"]["code"] == "not_found"


async def test_fail_if_active_does_not_clobber_a_completed_job(jobstore: JobStore) -> None:
    """Cancellation races the runner; a finished render must not become failed.

    DELETE reads the row, then writes. If the runner completes in between,
    an unconditional write flipped a completed job to failed and orphaned
    its file while telling the client the render had failed.
    """
    await jobstore.create(job_id="j1", model="p/m", prompt="x", size=None, seconds=None)
    await jobstore.update("j1", status="completed", file_id="f1", progress_pct=100)

    changed = await jobstore.fail_if_active("j1", "Cancelled by user")

    assert changed is False
    job = await jobstore.get("j1")
    assert job is not None
    assert job.status == "completed"
    assert job.file_id == "f1"


async def test_fail_if_active_transitions_an_active_job(jobstore: JobStore) -> None:
    await jobstore.create(job_id="j2", model="p/m", prompt="x", size=None, seconds=None)
    await jobstore.update("j2", status="in_progress")

    changed = await jobstore.fail_if_active("j2", "Cancelled by user")

    assert changed is True
    job = await jobstore.get("j2")
    assert job is not None
    assert job.status == "failed"
    assert job.error_message == "Cancelled by user"


async def test_first_writer_wins_on_the_error_message(jobstore: JobStore) -> None:
    """The runner's CancelledError handler must not overwrite the canceller."""
    await jobstore.create(job_id="j3", model="p/m", prompt="x", size=None, seconds=None)
    await jobstore.update("j3", status="in_progress")

    assert await jobstore.fail_if_active("j3", "Cancelled by user") is True
    assert await jobstore.fail_if_active("j3", "Job cancelled") is False

    job = await jobstore.get("j3")
    assert job is not None
    assert job.error_message == "Cancelled by user"


@respx.mock
@pytest.mark.asyncio
async def test_poll_interval_ramps_instead_of_staying_flat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A flat interval is wrong at both ends of ComfyUI's range.

    Fast enough for a 3-second image render is ~900 requests over a 15-minute
    video, at a web thread that is CPU-starved during generation — polling that
    hard slows the render it is waiting on. Slow enough for the video adds most
    of a second of dead time to every image.
    """
    slept: list[float] = []

    async def record(delay: float) -> None:
        slept.append(delay)

    monkeypatch.setattr(comfy_client_module.asyncio, "sleep", record)
    respx.get("http://comfy/history/ramp-id").mock(
        side_effect=[httpx.Response(200, json={}) for _ in range(8)]
        + [httpx.Response(200, json={"ramp-id": {"outputs": {}}})]
    )
    respx.get("http://comfy/queue").mock(
        return_value=httpx.Response(
            200, json={"queue_running": [[0, "ramp-id", {}, {}]], "queue_pending": []}
        )
    )

    client = ComfyUIClient(base_url="http://comfy", poll_interval_seconds=0.25)
    try:
        await client.poll_completion("ramp-id", timeout_seconds=600.0, max_interval=5.0)
    finally:
        await client.aclose()

    # First check comes quickly, so a fast workflow isn't left waiting.
    assert slept[0] == pytest.approx(0.25)
    # And the interval grows rather than staying put.
    assert slept == sorted(slept)
    assert slept[-1] > slept[0]
    # Eight polls in, a long render is already checked far less often — over
    # 15 minutes that is ~190 requests rather than ~900.
    assert slept[7] == pytest.approx(0.25 * 1.5**7)


@respx.mock
@pytest.mark.asyncio
async def test_poll_interval_is_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    """The ceiling has to stay well inside QUEUE_RECHECK_INTERVAL, or the
    dropped-prompt detection loses its cadence."""
    slept: list[float] = []

    async def record(delay: float) -> None:
        slept.append(delay)

    monkeypatch.setattr(comfy_client_module.asyncio, "sleep", record)
    respx.get("http://comfy/history/capped-id").mock(
        side_effect=[httpx.Response(200, json={}) for _ in range(40)]
        + [httpx.Response(200, json={"capped-id": {"outputs": {}}})]
    )
    respx.get("http://comfy/queue").mock(
        return_value=httpx.Response(
            200, json={"queue_running": [[0, "capped-id", {}, {}]], "queue_pending": []}
        )
    )

    client = ComfyUIClient(base_url="http://comfy", poll_interval_seconds=0.25)
    try:
        await client.poll_completion("capped-id", timeout_seconds=600.0, max_interval=5.0)
    finally:
        await client.aclose()

    assert max(slept) == pytest.approx(5.0)
    assert max(slept) < comfy_client_module.QUEUE_RECHECK_INTERVAL


@pytest.mark.asyncio
async def test_ceiling_never_outruns_a_deliberately_slow_start() -> None:
    """An operator who slows polling down shouldn't have the default ceiling
    speed it back up."""
    client = ComfyUIClient(base_url="http://comfy", poll_interval_seconds=20.0)
    try:
        assert client.max_poll_interval == 20.0
    finally:
        await client.aclose()
