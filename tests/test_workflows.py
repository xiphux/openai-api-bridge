"""ComfyUI workflow discovery + prepare_workflow correctness."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from openai_api_bridge.backends.comfyui.workflows import (
    prepare_workflow,
    scan_workflows,
    seconds_to_frames,
)
from openai_api_bridge.errors import WorkflowInvalid


def _write_pair(workflows_dir: Path, name: str, graph: dict, meta: dict) -> Path:
    j = workflows_dir / f"{name}.json"
    m = workflows_dir / f"{name}.meta.json"
    j.write_text(json.dumps(graph))
    m.write_text(json.dumps(meta))
    return j


def test_scan_skips_workflows_without_meta(tmp_path: Path) -> None:
    (tmp_path / "orphan.json").write_text(json.dumps({"1": {"class_type": "Foo"}}))
    out = scan_workflows(tmp_path)
    assert out == {}


def test_scan_skips_meta_missing_required_field(tmp_path: Path) -> None:
    _write_pair(
        tmp_path,
        "noprompt",
        graph={"1": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}}},
        meta={"display_name": "no prompt node declared"},
    )
    assert scan_workflows(tmp_path) == {}


def test_scan_handles_filenames_with_dots(tmp_path: Path) -> None:
    """Important regression: Path.with_suffix would mangle "Flux.2 Klein.json"."""
    _write_pair(
        tmp_path,
        "Flux.2 Klein",
        graph={"1": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}}},
        meta={"positive_prompt_node": "1"},
    )
    out = scan_workflows(tmp_path)
    assert "flux-2-klein" in out
    assert out["flux-2-klein"].display_name == "Flux.2 Klein"


def test_autodetect_video_output(tmp_path: Path) -> None:
    _write_pair(
        tmp_path,
        "vid",
        graph={
            "1": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}},
            "2": {"class_type": "VHS_VideoCombine", "inputs": {}},
        },
        meta={"positive_prompt_node": "1"},
    )
    rec = scan_workflows(tmp_path)["vid"]
    assert rec.output_type == "video"


def test_autodetect_image_output(tmp_path: Path) -> None:
    _write_pair(
        tmp_path,
        "img",
        graph={
            "1": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}},
            "2": {"class_type": "SaveImage", "inputs": {}},
        },
        meta={"positive_prompt_node": "1"},
    )
    rec = scan_workflows(tmp_path)["img"]
    assert rec.output_type == "image"


def test_explicit_output_type_overrides_autodetect(tmp_path: Path) -> None:
    _write_pair(
        tmp_path,
        "force",
        graph={
            "1": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}},
            "2": {"class_type": "VHS_VideoCombine", "inputs": {}},
        },
        meta={"positive_prompt_node": "1", "output_type": "image"},
    )
    rec = scan_workflows(tmp_path)["force"]
    assert rec.output_type == "image"


def test_prepare_injects_prompt_into_named_field(tmp_path: Path) -> None:
    _write_pair(
        tmp_path,
        "p",
        graph={
            "10": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}},
        },
        meta={"positive_prompt_node": "10"},
    )
    rec = scan_workflows(tmp_path)["p"]
    out = prepare_workflow(rec, prompt_text="HELLO")
    assert out["10"]["inputs"]["text"] == "HELLO"


def test_prepare_uses_custom_field_name(tmp_path: Path) -> None:
    _write_pair(
        tmp_path,
        "c",
        graph={"42": {"class_type": "Foo", "inputs": {"prompt": ""}}},
        meta={"positive_prompt_node": "42", "positive_prompt_field": "prompt"},
    )
    rec = scan_workflows(tmp_path)["c"]
    out = prepare_workflow(rec, prompt_text="X")
    assert out["42"]["inputs"]["prompt"] == "X"


def test_prepare_raises_on_missing_prompt_node(tmp_path: Path) -> None:
    _write_pair(
        tmp_path,
        "m",
        graph={"99": {"class_type": "X", "inputs": {}}},
        meta={"positive_prompt_node": "missing-id"},
    )
    rec = scan_workflows(tmp_path)["m"]
    with pytest.raises(WorkflowInvalid):
        prepare_workflow(rec, prompt_text="X")


def test_prepare_injects_dimensions(tmp_path: Path) -> None:
    _write_pair(
        tmp_path,
        "d",
        graph={
            "1": {"class_type": "T", "inputs": {"text": ""}},
            "5": {"class_type": "EmptyLatent", "inputs": {"width": 512, "height": 512}},
        },
        meta={
            "positive_prompt_node": "1",
            "dimensions_node": "5",
        },
    )
    rec = scan_workflows(tmp_path)["d"]
    out = prepare_workflow(rec, prompt_text="x", width=1024, height=768)
    assert out["5"]["inputs"]["width"] == 1024
    assert out["5"]["inputs"]["height"] == 768


def test_prepare_respects_custom_dim_field_names(tmp_path: Path) -> None:
    _write_pair(
        tmp_path,
        "d2",
        graph={
            "1": {"class_type": "T", "inputs": {"text": ""}},
            "5": {"class_type": "Custom", "inputs": {"resW": 0, "resH": 0}},
        },
        meta={
            "positive_prompt_node": "1",
            "dimensions_node": "5",
            "width_field": "resW",
            "height_field": "resH",
        },
    )
    rec = scan_workflows(tmp_path)["d2"]
    out = prepare_workflow(rec, prompt_text="x", width=2048, height=1024)
    assert out["5"]["inputs"]["resW"] == 2048
    assert out["5"]["inputs"]["resH"] == 1024


def test_prepare_zero_or_none_dims_leaves_workflow_default(tmp_path: Path) -> None:
    _write_pair(
        tmp_path,
        "d3",
        graph={
            "1": {"class_type": "T", "inputs": {"text": ""}},
            "5": {"class_type": "X", "inputs": {"width": 512, "height": 512}},
        },
        meta={"positive_prompt_node": "1", "dimensions_node": "5"},
    )
    rec = scan_workflows(tmp_path)["d3"]
    out = prepare_workflow(rec, prompt_text="x", width=None, height=None)
    assert out["5"]["inputs"]["width"] == 512  # untouched
    assert out["5"]["inputs"]["height"] == 512


def test_prepare_injects_image_filenames_singular(tmp_path: Path) -> None:
    _write_pair(
        tmp_path,
        "i",
        graph={
            "1": {"class_type": "T", "inputs": {"text": ""}},
            "7": {"class_type": "LoadImage", "inputs": {"image": ""}},
        },
        meta={
            "positive_prompt_node": "1",
            "image_inputs": [{"node": "7", "field": "image"}],
        },
    )
    rec = scan_workflows(tmp_path)["i"]
    out = prepare_workflow(rec, prompt_text="x", image_filenames=["sub_a.png"])
    assert out["7"]["inputs"]["image"] == "sub_a.png"


def test_prepare_image_inputs_multiple_serializes_remaining(tmp_path: Path) -> None:
    _write_pair(
        tmp_path,
        "i2",
        graph={
            "1": {"class_type": "T", "inputs": {"text": ""}},
            "7": {"class_type": "LoadMany", "inputs": {"images": ""}},
        },
        meta={
            "positive_prompt_node": "1",
            "image_inputs": [{"node": "7", "field": "images", "multiple": True}],
        },
    )
    rec = scan_workflows(tmp_path)["i2"]
    out = prepare_workflow(rec, prompt_text="x", image_filenames=["a.png", "b.png"])
    assert json.loads(out["7"]["inputs"]["images"]) == ["a.png", "b.png"]


def test_prepare_seeds_only_named_nodes_when_specified(tmp_path: Path) -> None:
    _write_pair(
        tmp_path,
        "s",
        graph={
            "1": {"class_type": "T", "inputs": {"text": ""}},
            "11": {"class_type": "KSampler", "inputs": {"seed": 1}},
            "12": {"class_type": "KSampler", "inputs": {"seed": 2}},
        },
        meta={"positive_prompt_node": "1", "seed_nodes": ["11"]},
    )
    rec = scan_workflows(tmp_path)["s"]
    rng = random.Random(0)
    out = prepare_workflow(rec, prompt_text="x", rng=rng)
    assert out["11"]["inputs"]["seed"] != 1  # randomized
    assert out["12"]["inputs"]["seed"] == 2  # left alone


def test_prepare_seeds_all_nodes_when_no_seed_nodes_meta(tmp_path: Path) -> None:
    _write_pair(
        tmp_path,
        "s2",
        graph={
            "1": {"class_type": "T", "inputs": {"text": ""}},
            "11": {"class_type": "KSampler", "inputs": {"seed": 1}},
            "12": {"class_type": "KSampler", "inputs": {"noise_seed": 2}},
        },
        meta={"positive_prompt_node": "1"},
    )
    rec = scan_workflows(tmp_path)["s2"]
    rng = random.Random(42)
    out = prepare_workflow(rec, prompt_text="x", rng=rng)
    assert out["11"]["inputs"]["seed"] != 1
    assert out["12"]["inputs"]["noise_seed"] != 2


def test_prepare_injects_length_for_video(tmp_path: Path) -> None:
    _write_pair(
        tmp_path,
        "v",
        graph={
            "1": {"class_type": "T", "inputs": {"text": ""}},
            "8": {"class_type": "FrameCount", "inputs": {"value": 25}},
        },
        meta={
            "positive_prompt_node": "1",
            "length_node": "8",
        },
    )
    rec = scan_workflows(tmp_path)["v"]
    out = prepare_workflow(rec, prompt_text="x", length=120)
    assert out["8"]["inputs"]["value"] == 120


def test_seconds_to_frames_with_fps_meta() -> None:
    assert seconds_to_frames(2.0, {"fps": 24}) == 48
    assert seconds_to_frames(0.5, {"fps": 30}) == 15


def test_seconds_to_frames_without_fps_meta() -> None:
    assert seconds_to_frames(5.0, {}) is None
    assert seconds_to_frames(None, {"fps": 30}) is None
    assert seconds_to_frames(0, {"fps": 30}) is None
