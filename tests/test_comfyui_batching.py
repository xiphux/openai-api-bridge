"""ComfyUI batching and workflow-discovery caching.

Two properties that only show up under repetition:

* ``n>1`` queues every run before waiting on any of them. Running them
  end-to-end one at a time meant the caller waited for the sum of n full
  generations inside a single synchronous request, and gave up the
  pipelining ComfyUI's queue exists to provide.
* ``cache_workflows = false`` re-reads the directory only when it has
  actually changed, rather than reparsing every meta file per request.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any

import pytest

from openai_api_bridge.backends.comfyui.adapter import ComfyUIBackend
from openai_api_bridge.config import ComfyUIProviderConfig
from openai_api_bridge.errors import UpstreamError


@pytest.fixture
def workflows_dir(tmp_path: Path) -> Path:
    d = tmp_path / "workflows"
    d.mkdir()
    (d / "wf.json").write_text(
        json.dumps({"3": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}}})
    )
    (d / "wf.meta.json").write_text(
        json.dumps({"positive_prompt_node": "3", "output_type": "image"})
    )
    return d


class _RecordingClient:
    """Stands in for ComfyUIClient, recording the order of operations."""

    def __init__(self) -> None:
        self.calls: list[str] = []
        self._next = 0

    async def submit_prompt(self, workflow: dict[str, Any]) -> str:
        self._next += 1
        prompt_id = f"p{self._next}"
        self.calls.append(f"submit:{prompt_id}")
        return prompt_id

    async def poll_completion(self, prompt_id: str, *, timeout_seconds: float) -> dict[str, Any]:
        self.calls.append(f"poll:{prompt_id}")
        return {"outputs": {}}

    async def retrieve_media(
        self, history_entry: dict[str, Any], *, output_type: str
    ) -> tuple[bytes, str]:
        return b"bytes", "image/png"

    async def delete_queued(self, prompt_ids: list[str]) -> None:
        self.calls.append(f"discard:{','.join(prompt_ids)}")

    async def aclose(self) -> None:
        pass


def _backend(workflows_dir: Path) -> tuple[ComfyUIBackend, _RecordingClient]:
    cfg = ComfyUIProviderConfig(backend="comfyui", id="c", workflows_dir=workflows_dir)
    backend = ComfyUIBackend(cfg)
    recorder = _RecordingClient()
    backend.client = recorder  # type: ignore[assignment]
    return backend, recorder


async def test_generate_image_submits_all_runs_before_collecting(workflows_dir: Path) -> None:
    backend, recorder = _backend(workflows_dir)

    assets = await backend.generate_image(model_slug="wf", prompt="a cat", n=3)

    assert len(assets) == 3
    submits = [c for c in recorder.calls if c.startswith("submit:")]
    polls = [c for c in recorder.calls if c.startswith("poll:")]
    assert len(submits) == 3
    assert len(polls) == 3
    # Every submit must precede the first poll — that's the pipelining.
    assert recorder.calls.index(polls[0]) > recorder.calls.index(submits[-1])


async def test_edit_image_submits_all_runs_before_collecting(
    workflows_dir: Path, tmp_path: Path
) -> None:
    (workflows_dir / "wfe.json").write_text(
        json.dumps(
            {
                "3": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}},
                "5": {"class_type": "LoadImage", "inputs": {"image": ""}},
            }
        )
    )
    (workflows_dir / "wfe.meta.json").write_text(
        json.dumps(
            {
                "positive_prompt_node": "3",
                "output_type": "image",
                "image_inputs": [{"node": "5", "field": "image"}],
            }
        )
    )
    backend, recorder = _backend(workflows_dir)

    async def _upload(data: bytes, content_type: str) -> str:
        return "uploaded.png"

    recorder.upload_image = _upload  # type: ignore[attr-defined]

    from openai_api_bridge.backends.base import InputImage

    assets = await backend.edit_image(
        model_slug="wfe",
        prompt="make it blue",
        images=[InputImage(data=b"img", content_type="image/png")],
        n=2,
    )

    assert len(assets) == 2
    submits = [c for c in recorder.calls if c.startswith("submit:")]
    polls = [c for c in recorder.calls if c.startswith("poll:")]
    assert len(submits) == 2
    assert recorder.calls.index(polls[0]) > recorder.calls.index(submits[-1])


async def test_single_run_still_works(workflows_dir: Path) -> None:
    backend, recorder = _backend(workflows_dir)

    assets = await backend.generate_image(model_slug="wf", prompt="a cat", n=1)

    assert len(assets) == 1
    assert recorder.calls == ["submit:p1", "poll:p1"]


async def test_workflows_are_not_rescanned_when_nothing_changed(
    workflows_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """cache_workflows=false rescanned on every request; now it stats instead."""
    from openai_api_bridge.backends.comfyui import adapter as adapter_module

    scans = 0
    real_scan = adapter_module.scan_workflows

    def counting_scan(d: Path):
        nonlocal scans
        scans += 1
        return real_scan(d)

    monkeypatch.setattr(adapter_module, "scan_workflows", counting_scan)

    cfg = ComfyUIProviderConfig(
        backend="comfyui", id="c", workflows_dir=workflows_dir, cache_workflows=False
    )
    backend = ComfyUIBackend(cfg)
    backend.client = _RecordingClient()  # type: ignore[assignment]

    await backend.list_models()
    await backend.list_models()
    await backend.list_models()

    assert scans == 1, f"directory was rescanned {scans} times with no changes"


async def test_workflows_are_rescanned_when_a_meta_file_changes(
    workflows_dir: Path,
) -> None:
    """The point of cache_workflows=false is picking up edits without a restart."""
    cfg = ComfyUIProviderConfig(
        backend="comfyui", id="c", workflows_dir=workflows_dir, cache_workflows=False
    )
    backend = ComfyUIBackend(cfg)
    backend.client = _RecordingClient()  # type: ignore[assignment]

    before = await backend.list_models()
    assert [m.display_name for m in before] == ["wf"]

    (workflows_dir / "wf.meta.json").write_text(
        json.dumps({"positive_prompt_node": "3", "output_type": "image", "display_name": "Renamed"})
    )

    after = await backend.list_models()
    assert [m.display_name for m in after] == ["Renamed"]


async def test_cache_workflows_true_does_not_rescan(workflows_dir: Path) -> None:
    cfg = ComfyUIProviderConfig(
        backend="comfyui", id="c", workflows_dir=workflows_dir, cache_workflows=True
    )
    backend = ComfyUIBackend(cfg)
    backend.client = _RecordingClient()  # type: ignore[assignment]

    await backend.list_models()
    (workflows_dir / "wf.meta.json").write_text(
        json.dumps({"positive_prompt_node": "3", "output_type": "image", "display_name": "Renamed"})
    )
    after = await backend.list_models()

    assert [m.display_name for m in after] == ["wf"]


async def test_batch_budgets_the_whole_queue_not_a_single_run(workflows_dir: Path) -> None:
    """Collectors all start their clock at submit time, but runs execute serially.

    Giving each collector a single run's budget times out a healthy batch on
    queue position alone: at n=4 the last run waits out three others first.
    """
    cfg = ComfyUIProviderConfig(
        backend="comfyui",
        id="c",
        workflows_dir=workflows_dir,
        poll_timeout_image_seconds=300.0,
    )
    backend = ComfyUIBackend(cfg)
    recorder = _RecordingClient()
    seen: list[float] = []

    async def poll(prompt_id: str, *, timeout_seconds: float) -> dict[str, Any]:
        seen.append(timeout_seconds)
        return {"outputs": {}}

    recorder.poll_completion = poll  # type: ignore[assignment]
    backend.client = recorder  # type: ignore[assignment]

    await backend.generate_image(model_slug="wf", prompt="a cat", n=4)

    assert seen == [1200.0] * 4, "each collector should budget for the full queue"


async def test_single_run_keeps_the_plain_per_run_budget(workflows_dir: Path) -> None:
    cfg = ComfyUIProviderConfig(
        backend="comfyui",
        id="c",
        workflows_dir=workflows_dir,
        poll_timeout_image_seconds=300.0,
    )
    backend = ComfyUIBackend(cfg)
    recorder = _RecordingClient()
    seen: list[float] = []

    async def poll(prompt_id: str, *, timeout_seconds: float) -> dict[str, Any]:
        seen.append(timeout_seconds)
        return {"outputs": {}}

    recorder.poll_completion = poll  # type: ignore[assignment]
    backend.client = recorder  # type: ignore[assignment]

    await backend.generate_image(model_slug="wf", prompt="a cat", n=1)

    assert seen == [300.0]


async def test_one_failing_collector_cancels_its_siblings(workflows_dir: Path) -> None:
    """A failed prompt must cancel the rest, not wait them out.

    gather surfaces the first error immediately but leaves the other tasks
    running, so they keep polling ComfyUI and buffering media for a request
    that has already failed. return_exceptions=True does not fix this — it
    cancels nothing and merely makes the client wait out the whole batch
    budget for an error it could have had at once.
    """
    backend, recorder = _backend(workflows_dir)
    finished: list[str] = []

    async def poll(prompt_id: str, *, timeout_seconds: float) -> dict[str, Any]:
        if prompt_id == "p2":
            raise UpstreamError("ComfyUI dropped the prompt")
        await asyncio.sleep(5.0)  # a sibling still mid-poll
        finished.append(prompt_id)
        return {"outputs": {}}

    recorder.poll_completion = poll  # type: ignore[assignment]

    start = time.perf_counter()
    with pytest.raises(UpstreamError):
        await backend.generate_image(model_slug="wf", prompt="a cat", n=3)
    elapsed = time.perf_counter() - start

    assert finished == [], "siblings should have been cancelled, not awaited"
    assert elapsed < 1.0, f"failed fast? took {elapsed:.2f}s waiting on siblings"


async def test_failed_submit_reports_the_orphaned_prompts(
    workflows_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """A mid-batch submit failure leaves queued prompts rendering uncollected."""
    backend, recorder = _backend(workflows_dir)
    calls = 0

    async def submit(workflow: dict[str, Any]) -> str:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise UpstreamError("ComfyUI rejected the prompt")
        return f"p{calls}"

    recorder.submit_prompt = submit  # type: ignore[assignment]

    with caplog.at_level(logging.WARNING), pytest.raises(UpstreamError):
        await backend.generate_image(model_slug="wf", prompt="a cat", n=4)

    assert "Abandoning 2 queued ComfyUI prompt(s)" in caplog.text
    assert "p1, p2" in caplog.text
    # and they're handed back rather than left to render for nobody
    assert "discard:p1,p2" in recorder.calls


async def test_cancelled_siblings_have_their_prompts_recalled(workflows_dir: Path) -> None:
    """Cancelling siblings promptly must not silently strand their prompts.

    The submit path already recalled orphans; cancellation moved the same
    orphan class into the collect path, which is the likelier route.
    """
    backend, recorder = _backend(workflows_dir)

    async def poll(prompt_id: str, *, timeout_seconds: float) -> dict[str, Any]:
        if prompt_id == "p2":
            raise UpstreamError("ComfyUI dropped the prompt")
        await asyncio.sleep(5.0)
        return {"outputs": {}}

    recorder.poll_completion = poll  # type: ignore[assignment]

    with pytest.raises(UpstreamError):
        await backend.generate_image(model_slug="wf", prompt="a cat", n=3)

    discards = [c for c in recorder.calls if c.startswith("discard:")]
    assert discards, "cancelled collectors left their prompts queued upstream"
    recalled = discards[0].removeprefix("discard:").split(",")
    # p2 failed and p1/p3 were cancelled — none produced a result.
    assert sorted(recalled) == ["p1", "p2", "p3"]


async def test_successfully_collected_prompts_are_not_recalled(workflows_dir: Path) -> None:
    """Only prompts that never produced a result should be handed back."""
    backend, recorder = _backend(workflows_dir)
    done = asyncio.Event()

    async def poll(prompt_id: str, *, timeout_seconds: float) -> dict[str, Any]:
        if prompt_id == "p1":
            done.set()
            return {"outputs": {}}
        await done.wait()
        raise UpstreamError("ComfyUI dropped the prompt")

    recorder.poll_completion = poll  # type: ignore[assignment]

    with pytest.raises(UpstreamError):
        await backend.generate_image(model_slug="wf", prompt="a cat", n=2)

    discards = [c for c in recorder.calls if c.startswith("discard:")]
    assert discards == ["discard:p2"], f"p1 completed and shouldn't be recalled: {discards}"


async def test_cancelling_a_single_run_recalls_its_prompt(workflows_dir: Path) -> None:
    """A cancelled video job shouldn't leave ComfyUI rendering for nobody.

    DELETE /v1/videos/{id} cancels the runner task, which cancels us here.
    """
    backend, recorder = _backend(workflows_dir)
    polling = asyncio.Event()

    async def poll(prompt_id: str, *, timeout_seconds: float) -> dict[str, Any]:
        polling.set()
        await asyncio.sleep(30.0)
        return {"outputs": {}}

    recorder.poll_completion = poll  # type: ignore[assignment]

    task = asyncio.create_task(backend.generate_image(model_slug="wf", prompt="a cat", n=1))
    await polling.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert "discard:p1" in recorder.calls
