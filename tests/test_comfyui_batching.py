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

    async def poll_completion(
        self,
        prompt_id: str,
        *,
        timeout_seconds: float,
        max_interval: float | None = None,
    ) -> dict[str, Any]:
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

    async def poll(
        prompt_id: str, *, timeout_seconds: float, max_interval: float | None = None
    ) -> dict[str, Any]:
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

    async def poll(
        prompt_id: str, *, timeout_seconds: float, max_interval: float | None = None
    ) -> dict[str, Any]:
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

    async def poll(
        prompt_id: str, *, timeout_seconds: float, max_interval: float | None = None
    ) -> dict[str, Any]:
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

    async def poll(
        prompt_id: str, *, timeout_seconds: float, max_interval: float | None = None
    ) -> dict[str, Any]:
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

    async def poll(
        prompt_id: str, *, timeout_seconds: float, max_interval: float | None = None
    ) -> dict[str, Any]:
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

    async def poll(
        prompt_id: str, *, timeout_seconds: float, max_interval: float | None = None
    ) -> dict[str, Any]:
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


async def test_discard_is_bounded_when_comfyui_hangs(workflows_dir: Path) -> None:
    """A cancel must not wait on an unresponsive ComfyUI.

    The recall runs while unwinding, and DELETE /v1/videos/{id} doesn't
    return until it finishes.
    """
    backend, recorder = _backend(workflows_dir)

    async def hanging_delete(prompt_ids: list[str]) -> None:
        await asyncio.sleep(30.0)

    async def poll(
        prompt_id: str, *, timeout_seconds: float, max_interval: float | None = None
    ) -> dict[str, Any]:
        raise UpstreamError("ComfyUI dropped the prompt")

    recorder.delete_queued = hanging_delete  # type: ignore[assignment]
    recorder.poll_completion = poll  # type: ignore[assignment]

    monkeypatched = 0.05
    from openai_api_bridge.backends.comfyui import adapter as adapter_module

    original = adapter_module._QUEUE_DISCARD_TIMEOUT_S
    adapter_module._QUEUE_DISCARD_TIMEOUT_S = monkeypatched
    try:
        start = time.perf_counter()
        with pytest.raises(UpstreamError):
            await backend.generate_image(model_slug="wf", prompt="a cat", n=1)
        elapsed = time.perf_counter() - start
    finally:
        adapter_module._QUEUE_DISCARD_TIMEOUT_S = original

    assert elapsed < 1.0, f"a hanging discard blocked the caller for {elapsed:.2f}s"


def test_discard_timeout_stays_short() -> None:
    """Pins the value, not just the mechanism — a silent revert to 10s is the
    regression this guards, and it's invisible in behaviour tests."""
    from openai_api_bridge.backends.comfyui import adapter as adapter_module

    assert adapter_module._QUEUE_DISCARD_TIMEOUT_S <= 2.0


async def test_batch_reads_the_graph_once_not_once_per_run(
    workflows_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The graph read used to sit inside prepare_workflow, so a batch did a
    blocking read and parse of a 50-200KB file per run, on the event loop the
    whole bridge shares."""
    from openai_api_bridge.backends.comfyui import adapter as adapter_module

    reads: list[str] = []
    real = adapter_module.read_graph_text

    def counting(record: Any) -> str:
        reads.append(record.slug)
        return real(record)

    monkeypatch.setattr(adapter_module, "read_graph_text", counting)

    backend, _ = _backend(workflows_dir)
    await backend.generate_image(model_slug="wf", prompt="a cat", n=4)

    assert reads == ["wf"], f"expected one read for the batch, got {len(reads)}"


async def test_each_run_in_a_batch_gets_its_own_graph(workflows_dir: Path) -> None:
    """Sharing one parsed dict across runs would make every submit mutate the
    same object — the seeds would collapse to whatever the last run rolled,
    and ComfyUI's execution cache would return identical outputs."""
    (workflows_dir / "wf.json").write_text(
        json.dumps(
            {
                "3": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}},
                "4": {"class_type": "KSampler", "inputs": {"seed": 0}},
            }
        )
    )
    backend, _ = _backend(workflows_dir)

    submitted: list[dict[str, Any]] = []

    async def capture(workflow: dict[str, Any]) -> str:
        submitted.append(workflow)
        return f"p{len(submitted)}"

    backend.client.submit_prompt = capture  # type: ignore[method-assign]

    await backend.generate_image(model_slug="wf", prompt="a cat", n=4)

    seeds = [w["4"]["inputs"]["seed"] for w in submitted]
    assert len(set(seeds)) == 4, f"runs shared a graph — seeds collapsed to {seeds}"


async def test_edited_workflow_takes_effect_on_the_next_generation(
    workflows_dir: Path,
) -> None:
    """The read moved off the loop, but it must stay per request — the
    documented behaviour is that saving an edited workflow applies without a
    restart."""
    backend, _ = _backend(workflows_dir)

    submitted: list[dict[str, Any]] = []

    async def capture(workflow: dict[str, Any]) -> str:
        submitted.append(workflow)
        return "p1"

    backend.client.submit_prompt = capture  # type: ignore[method-assign]

    await backend.generate_image(model_slug="wf", prompt="a cat", n=1)
    assert "9" not in submitted[0]

    (workflows_dir / "wf.json").write_text(
        json.dumps(
            {
                "3": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}},
                "9": {"class_type": "AddedLater", "inputs": {}},
            }
        )
    )
    await backend.generate_image(model_slug="wf", prompt="a cat", n=1)
    assert "9" in submitted[1]


@pytest.mark.parametrize(
    ("output_type", "expected"),
    [("image", 1.0), ("video", 5.0)],
)
async def test_poll_ceiling_differs_by_output_type(
    workflows_dir: Path, output_type: str, expected: float
) -> None:
    """One ceiling cannot serve both paths.

    An image is collected inside the caller's synchronous
    POST /v1/images/generations, so every second the ramp adds between
    "ComfyUI finished" and "we noticed" is a second they sit through. A video
    is collected by a background job the client polls on its own cadence,
    where that lag is free and the request volume over a 15-minute render is
    what actually matters.
    """
    (workflows_dir / "wf.meta.json").write_text(
        json.dumps({"positive_prompt_node": "3", "output_type": output_type})
    )
    backend, _ = _backend(workflows_dir)
    record = (await backend._ensure_workflows())["wf"]

    assert backend._max_poll_interval_for(record) == expected


async def test_image_polling_never_eases_out_past_its_ceiling(workflows_dir: Path) -> None:
    """The ceiling the adapter picks is the one the client actually applies."""
    backend, recorder = _backend(workflows_dir)
    seen: dict[str, float | None] = {}

    async def capture(
        prompt_id: str, *, timeout_seconds: float, max_interval: float | None = None
    ) -> dict[str, Any]:
        seen["max_interval"] = max_interval
        return {"outputs": {}}

    recorder.poll_completion = capture  # type: ignore[assignment]

    await backend.generate_image(model_slug="wf", prompt="a cat", n=1)

    assert seen["max_interval"] == 1.0
