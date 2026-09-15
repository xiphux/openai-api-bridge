"""ComfyUI video generation: the input_reference and seconds paths.

``generate_video`` is how ``POST /v1/videos`` reaches a ComfyUI workflow, and
only its rejection of an image workflow was tested. These pin what happens to a
reference image (uploaded and wired into the workflow's declared image node,
refused when required and missing, ignored when the workflow takes none) and to
``seconds`` (turned into a frame count via the meta's ``fps``).

Uses a recording stand-in for ``ComfyUIClient``, as test_comfyui_batching.py
does, so the assertions are on what the adapter asked ComfyUI to do.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from openai_api_bridge.backends.comfyui.adapter import ComfyUIBackend
from openai_api_bridge.config import ComfyUIProviderConfig
from openai_api_bridge.errors import ImageRequired

_GRAPH = {
    "1": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}},
    "7": {"class_type": "LoadImage", "inputs": {"image": "placeholder.png"}},
    "8": {"class_type": "PrimitiveInt", "inputs": {"value": 97}},
    "9": {"class_type": "VHS_VideoCombine", "inputs": {}},
}
_MP4 = b"\x00\x00\x00\x18ftypmp42"


def _write(d: Path, slug: str, meta: dict[str, Any]) -> None:
    (d / f"{slug}.json").write_text(json.dumps(_GRAPH))
    (d / f"{slug}.meta.json").write_text(json.dumps({"positive_prompt_node": "1", **meta}))


@pytest.fixture
def workflows_dir(tmp_path: Path) -> Path:
    d = tmp_path / "workflows"
    d.mkdir()
    image_node = {"image_inputs": [{"node": "7", "field": "image"}]}
    length = {"length_node": "8", "fps": 24}
    _write(d, "i2v-required", {**image_node, "image_required": True, **length})
    _write(d, "i2v-optional", {**image_node, **length})
    _write(d, "t2v", length)
    return d


class _RecordingClient:
    def __init__(self) -> None:
        self.uploads: list[tuple[bytes, str]] = []
        self.submitted: list[dict[str, Any]] = []

    async def upload_image(self, image_data: bytes, content_type: str) -> str:
        self.uploads.append((image_data, content_type))
        return f"uploaded-{len(self.uploads)}.png"

    async def submit_prompt(self, workflow: dict[str, Any]) -> str:
        self.submitted.append(workflow)
        return "prompt-1"

    async def poll_completion(self, prompt_id: str, **_: Any) -> dict[str, Any]:
        return {"outputs": {}}

    async def retrieve_media(self, history_entry: dict[str, Any], **_: Any) -> tuple[bytes, str]:
        return _MP4, "video/mp4"

    async def delete_queued(self, prompt_ids: list[str]) -> None:
        pass

    async def aclose(self) -> None:
        pass


def _backend(workflows_dir: Path) -> tuple[ComfyUIBackend, _RecordingClient]:
    cfg = ComfyUIProviderConfig(backend="comfyui", id="c", workflows_dir=workflows_dir)
    backend = ComfyUIBackend(cfg)
    recorder = _RecordingClient()
    backend.client = recorder  # type: ignore[assignment]
    return backend, recorder


async def test_reference_image_is_uploaded_and_wired_into_the_image_node(
    workflows_dir: Path,
) -> None:
    backend, recorder = _backend(workflows_dir)
    upstream_ids: list[str] = []

    async def on_upstream_id(upstream_id: str) -> None:
        upstream_ids.append(upstream_id)

    asset = await backend.generate_video(
        model_slug="i2v-required",
        prompt="the cat walks",
        seconds=2,
        input_reference=b"JPEGDATA",
        input_reference_content_type="image/jpeg",
        on_upstream_id=on_upstream_id,
    )

    assert (asset.data, asset.content_type, asset.kind) == (_MP4, "video/mp4", "video")
    assert recorder.uploads == [(b"JPEGDATA", "image/jpeg")]
    (workflow,) = recorder.submitted
    assert workflow["1"]["inputs"]["text"] == "the cat walks"
    assert workflow["7"]["inputs"]["image"] == "uploaded-1.png"
    assert workflow["8"]["inputs"]["value"] == 48  # 2s x 24fps
    assert upstream_ids == ["prompt-1"]


async def test_reference_content_type_defaults_to_png(workflows_dir: Path) -> None:
    backend, recorder = _backend(workflows_dir)

    await backend.generate_video(model_slug="i2v-optional", prompt="p", input_reference=b"PNGDATA")

    assert recorder.uploads == [(b"PNGDATA", "image/png")]


async def test_a_required_reference_that_is_missing_is_refused_before_submitting(
    workflows_dir: Path,
) -> None:
    backend, recorder = _backend(workflows_dir)

    with pytest.raises(ImageRequired) as exc:
        await backend.generate_video(model_slug="i2v-required", prompt="p")

    assert exc.value.param == "input_reference"
    assert recorder.uploads == []
    assert recorder.submitted == []


async def test_an_optional_reference_can_be_omitted(workflows_dir: Path) -> None:
    backend, recorder = _backend(workflows_dir)

    await backend.generate_video(model_slug="i2v-optional", prompt="p")

    assert recorder.uploads == []
    (workflow,) = recorder.submitted
    assert workflow["7"]["inputs"]["image"] == "placeholder.png"  # left as authored


async def test_a_workflow_without_image_inputs_ignores_a_reference(workflows_dir: Path) -> None:
    """Not an error: the reference is dropped, and nothing is uploaded that
    ComfyUI would then keep for no workflow to read."""
    backend, recorder = _backend(workflows_dir)

    await backend.generate_video(model_slug="t2v", prompt="p", input_reference=b"PNGDATA")

    assert recorder.uploads == []
    assert len(recorder.submitted) == 1


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (None, 97),  # no seconds: the workflow's baked-in length stands
        (0, 97),  # non-positive: same
        (0.01, 1),  # never rounds down to zero frames
        (4.5, 108),
    ],
)
async def test_seconds_become_a_frame_count_only_when_meaningful(
    workflows_dir: Path, seconds: float | None, expected: int
) -> None:
    backend, recorder = _backend(workflows_dir)

    await backend.generate_video(model_slug="t2v", prompt="p", seconds=seconds)

    (workflow,) = recorder.submitted
    assert workflow["8"]["inputs"]["value"] == expected
