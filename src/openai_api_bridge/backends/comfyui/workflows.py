"""ComfyUI workflow discovery + meta-driven workflow preparation.

Lifted (with light edits) from the existing Open WebUI pipe at
``open-webui-image-prompt-enhancer/comfyui_image_generation.py``.

A "workflow" on disk is a pair: ``{name}.json`` (the API-format graph) and
``{name}.meta.json`` (declarative bridge metadata — which node receives the
prompt, which receives images, dimensions/length nodes, etc.). See the meta
schema in the project README.
"""

from __future__ import annotations

import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...errors import UnsupportedOperation, WorkflowInvalid
from ...util.ids import slugify

log = logging.getLogger(__name__)

# Class types whose presence in a workflow indicates a video output. Auto-
# detection is only used when meta.json doesn't specify ``output_type``
# explicitly. Add new node types here as ComfyUI ecosystem evolves.
VIDEO_OUTPUT_CLASS_TYPES = frozenset({"SaveVideo", "VHS_VideoCombine"})

_SEED_FIELD_NAMES = frozenset({"seed", "noise_seed"})


@dataclass(frozen=True, slots=True)
class WorkflowRecord:
    slug: str
    json_path: Path
    meta: dict[str, Any]
    output_type: str  # "image" | "video"
    display_name: str


def scan_workflows(workflows_dir: Path) -> dict[str, WorkflowRecord]:
    """Discover workflow + meta pairs in ``workflows_dir``.

    Returns a map from slug → WorkflowRecord. Workflows without a companion
    .meta.json, or with an invalid meta, are skipped with a logged warning.
    """
    out: dict[str, WorkflowRecord] = {}
    if not workflows_dir.is_dir():
        log.warning("Workflows directory not found: %s", workflows_dir)
        return out

    for json_path in sorted(workflows_dir.glob("*.json")):
        if json_path.name.endswith(".meta.json"):
            continue
        # Use removesuffix() so filenames containing dots (e.g. "Flux.2 Klein.json")
        # are handled correctly. Path.with_suffix() would mangle them.
        meta_path = json_path.parent / (json_path.name.removesuffix(".json") + ".meta.json")
        if not meta_path.exists():
            log.debug("Skipping %s: no companion .meta.json", json_path.name)
            continue
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            log.warning("Skipping %s: meta read failed: %s", json_path.name, e)
            continue
        if "positive_prompt_node" not in meta:
            log.warning("Skipping %s: meta missing 'positive_prompt_node'", json_path.name)
            continue

        output_type = meta.get("output_type")
        if output_type not in ("image", "video"):
            output_type = _autodetect_output_type(json_path)
            if output_type is None:
                continue

        base = json_path.name.removesuffix(".json")
        slug = slugify(base)
        if not slug:
            log.warning("Skipping %s: name slugifies to empty string", json_path.name)
            continue
        display_name = meta.get("display_name") or base

        if slug in out:
            log.warning(
                "Duplicate workflow slug %r — keeping first, dropping %s",
                slug,
                json_path.name,
            )
            continue

        out[slug] = WorkflowRecord(
            slug=slug,
            json_path=json_path,
            meta=meta,
            output_type=output_type,
            display_name=display_name,
        )
        log.info(
            "Discovered workflow %r (%s) — output_type=%s",
            display_name,
            json_path.name,
            output_type,
        )
    return out


def _autodetect_output_type(json_path: Path) -> str | None:
    try:
        graph = json.loads(json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        log.warning("Skipping %s: workflow read failed: %s", json_path.name, e)
        return None
    has_video = any(
        isinstance(node, dict) and node.get("class_type") in VIDEO_OUTPUT_CLASS_TYPES
        for node in graph.values()
    )
    return "video" if has_video else "image"


def read_graph_text(record: WorkflowRecord) -> str:
    """Read a workflow's API-format graph off disk. **Blocking** — call via
    ``asyncio.to_thread``.

    Separate from :func:`prepare_workflow` so the read happens once per
    request rather than once per submit, and off the event loop the whole
    bridge shares. It used to be inline, which put a synchronous read plus a
    parse of a 50-200KB graph on the loop for every run in a batch.

    Still per request, so saving an edited workflow on disk takes effect on
    the next generation without restarting the bridge.
    """
    return record.json_path.read_text(encoding="utf-8")


def prepare_workflow(
    record: WorkflowRecord,
    graph_text: str,
    *,
    prompt_text: str,
    image_filenames: list[str] | None = None,
    width: int | None = None,
    height: int | None = None,
    length: int | None = None,
    rng: random.Random | None = None,
) -> dict[str, Any]:
    """Parse a workflow graph and inject this run's overrides.

    ``graph_text`` comes from :func:`read_graph_text`. Parsing per call is
    what gives each run in a batch its own mutable copy — measured at ~0.26ms
    for a 74KB graph, against ~2.1ms to ``deepcopy`` the parsed result, so
    re-parsing is both simpler and the faster of the two.

    ``rng`` is injectable for deterministic tests.
    """
    workflow: dict[str, Any] = json.loads(graph_text)
    meta = record.meta
    rng = rng or random.Random()

    # Positive prompt
    pos_node = meta["positive_prompt_node"]
    pos_field = meta.get("positive_prompt_field", "text")
    if pos_node not in workflow:
        raise WorkflowInvalid(
            f"Workflow {record.slug!r}: positive_prompt_node {pos_node!r} not present in graph"
        )
    workflow[pos_node].setdefault("inputs", {})[pos_field] = prompt_text

    # Image inputs (if any)
    image_inputs = meta.get("image_inputs", []) or []
    if image_inputs and image_filenames:
        remaining = list(image_filenames)
        for spec in image_inputs:
            node_id = spec.get("node")
            field = spec.get("field")
            fmt = spec.get("format", "filename")
            multiple = bool(spec.get("multiple", False))
            if not node_id or not field or node_id not in workflow:
                continue
            if multiple:
                if fmt == "filename":
                    workflow[node_id]["inputs"][field] = json.dumps(remaining)
                else:
                    workflow[node_id]["inputs"][field] = remaining
                remaining = []
            elif remaining:
                workflow[node_id]["inputs"][field] = remaining.pop(0)
        # Surplus images that no spec could consume would otherwise be
        # silently dropped (they're already uploaded to ComfyUI). Surface it
        # as an error instead — matches the edit_image contract that surplus
        # references error rather than vanish.
        if remaining:
            consumed = len(image_filenames) - len(remaining)
            raise UnsupportedOperation(
                f"Workflow {record.slug!r} accepts {consumed} image input(s) "
                f"but {len(image_filenames)} were supplied",
                param="image",
            )

    # Seed randomization. Without this, ComfyUI's execution cache will short-
    # circuit identical-input runs and return the same output every time.
    seed_nodes = meta.get("seed_nodes")
    nodes_to_seed: dict[str, Any] = (
        {nid: workflow[nid] for nid in seed_nodes if nid in workflow} if seed_nodes else workflow
    )
    for node in nodes_to_seed.values():
        if not isinstance(node, dict):
            continue
        inputs = node.get("inputs", {})
        for field_name in _SEED_FIELD_NAMES:
            if field_name in inputs and isinstance(inputs[field_name], int | float):
                inputs[field_name] = rng.randint(0, 2**32 - 1)

    # Dimensions
    dim_node = meta.get("dimensions_node")
    if dim_node and dim_node in workflow:
        if width and width > 0:
            workflow[dim_node]["inputs"][meta.get("width_field", "width")] = width
        if height and height > 0:
            workflow[dim_node]["inputs"][meta.get("height_field", "height")] = height

    # Length (video frame count)
    length_node = meta.get("length_node")
    if length_node and length_node in workflow and length and length > 0:
        workflow[length_node]["inputs"][meta.get("length_field", "value")] = length

    return workflow


def seconds_to_frames(seconds: float | None, meta: dict[str, Any]) -> int | None:
    """Translate OpenAI's ``seconds`` into a ComfyUI frame count if the workflow
    declares ``fps`` in its meta. Returns None when no translation is possible
    or desired (the workflow's baked-in default length is then preserved)."""
    if seconds is None or seconds <= 0:
        return None
    fps = meta.get("fps")
    if not fps:
        return None
    return max(1, round(seconds * float(fps)))
