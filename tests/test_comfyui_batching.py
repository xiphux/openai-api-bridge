"""ComfyUI ``n>1`` queues every run before waiting on any of them.

Running them end-to-end one at a time meant the caller waited for the sum
of n full generations inside a single synchronous request, and gave up the
pipelining ComfyUI's queue exists to provide.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from openai_api_bridge.backends.comfyui.adapter import ComfyUIBackend
from openai_api_bridge.config import ComfyUIProviderConfig


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
