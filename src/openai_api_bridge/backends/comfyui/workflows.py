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
from ...util.aspect import AspectRatio, parse_aspect_ratio, parse_aspect_ratios, snap_aspect_ratio
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
    # Resolved at scan time rather than read from ``meta`` at use sites: the
    # literals need parsing and validating, and doing it once per scan means one
    # warning per bad declaration instead of one per request.
    aspect_ratios: tuple[AspectRatio, ...] = ()
    # The canonical ratio the graph is saved with — what this workflow produces
    # when a request names none. Read from the graph, so it costs no meta field,
    # but it is captured at scan time and so goes stale with the rest of the
    # scan until ``cache_workflows`` lets it rescan.
    aspect_ratio_default: str | None = None


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

        # The graph is read here only when something needs it: autodetecting the
        # output type, or reading the aspect-ratio node's saved default. Most
        # workflows already pay for that read via autodetection.
        declared_type = meta.get("output_type")
        needs_autodetect = declared_type not in ("image", "video")
        wants_aspect = meta.get("aspect_ratio_node") is not None or "aspect_ratios" in meta
        graph = _read_graph(json_path) if needs_autodetect or wants_aspect else None

        if needs_autodetect:
            if graph is None:
                continue
            output_type = _output_type_of(graph)
        else:
            output_type = str(declared_type)

        aspect_ratios, aspect_ratio_default = _resolve_aspect_ratios(meta, graph, json_path.name)

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
            aspect_ratios=aspect_ratios,
            aspect_ratio_default=aspect_ratio_default,
        )
        log.info(
            "Discovered workflow %r (%s) — output_type=%s%s",
            display_name,
            json_path.name,
            output_type,
            f", {len(aspect_ratios)} aspect ratios" if aspect_ratios else "",
        )
    return out


def _read_graph(json_path: Path) -> dict[str, Any] | None:
    """Parse a workflow graph, or ``None`` when it can't be read."""
    try:
        parsed = json.loads(json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        log.warning("%s: workflow read failed: %s", json_path.name, e)
        return None
    if not isinstance(parsed, dict):
        log.warning("%s: workflow graph is not an object", json_path.name)
        return None
    return parsed


def _output_type_of(graph: dict[str, Any]) -> str:
    has_video = any(
        isinstance(node, dict) and node.get("class_type") in VIDEO_OUTPUT_CLASS_TYPES
        for node in graph.values()
    )
    return "video" if has_video else "image"


def _resolve_aspect_ratios(
    meta: dict[str, Any],
    graph: dict[str, Any] | None,
    filename: str,
) -> tuple[tuple[AspectRatio, ...], str | None]:
    """Validate a workflow's aspect-ratio declaration and read its default.

    Both halves are required: the node says where to inject, the list says what
    may be injected, and neither is derivable from the other — the graph holds
    only the *current* value, never the node's menu. Half a declaration is an
    operator mistake rather than a partial feature, so it warns and disables,
    which leaves the workflow generating at its baked-in ratio instead of
    advertising a menu it can't honour (or honouring values it never offered).
    """
    node_id = meta.get("aspect_ratio_node")
    raw = meta.get("aspect_ratios")
    if node_id is None and raw is None:
        return (), None

    if node_id is None or raw is None:
        missing = "aspect_ratio_node" if node_id is None else "aspect_ratios"
        log.warning(
            "%s: aspect ratios need both 'aspect_ratio_node' and 'aspect_ratios'; "
            "%r is missing, so the selector is disabled for this workflow",
            filename,
            missing,
        )
        return (), None

    options = parse_aspect_ratios(raw, where=filename)
    if not options:
        log.warning("%s: 'aspect_ratios' yielded no usable options; selector disabled", filename)
        return (), None

    if graph is None:
        # Everything below validates the declaration against the graph, so an
        # unreadable one leaves us advertising a menu we never checked — the exact
        # "advertised and silently inert" state the checks exist to prevent. The
        # workflow is unusable regardless: `read_graph_text` fails the same way at
        # generation time. (`_read_graph` has already said why it couldn't be read.)
        log.warning(
            "%s: declares aspect ratios but its graph could not be read, so the "
            "declaration can't be checked; selector disabled",
            filename,
        )
        return (), None

    # Nothing to inject into means the declaration is stale — a renumbered node,
    # usually, which is silent otherwise because ComfyUI ids aren't stable across
    # a re-export.
    if str(node_id) not in graph:
        log.warning(
            "%s: aspect_ratio_node %r is not in the graph; selector disabled",
            filename,
            node_id,
        )
        return (), None

    field = meta.get("aspect_ratio_field", "aspect_ratio")
    node = graph.get(str(node_id))
    inputs = node.get("inputs") if isinstance(node, dict) else None
    if not isinstance(inputs, dict) or field not in inputs:
        # A misdeclared field name is the one misconfiguration that would be wrong
        # on EVERY request rather than an unlucky few: injection writes through
        # `setdefault`, so naming an input the node doesn't have adds a key it
        # ignores. Every render would then use the graph's own ratio while the
        # response reported the request — and because the menu still advertises
        # fine, the only symptom is that picking a shape does nothing. Disable
        # instead, which at least fails the way a missing declaration does.
        log.warning(
            "%s: aspect_ratio_node %r has no %r input — check 'aspect_ratio_field'. "
            "Writing it would add a key the node ignores, so the selector is disabled "
            "rather than advertised and silently inert.",
            filename,
            node_id,
            field,
        )
        return (), None

    # Everything that could still disable the selector has now run, so from here
    # the ratio really does win — which is what makes this warning safe to emit.
    # Above the field check it was a lie: a workflow with both knobs AND a typo'd
    # field name got told the ratio was honoured and `size` ignored, when the
    # selector was about to be disabled and `size` was about to be honoured.
    #
    # An aspect-ratio node computes the dimensions the graph then uses, so a
    # workflow declaring both knobs has two writers for one value. Prefer the
    # ratio: it is the newer, more specific declaration, and it is the one a
    # client is being told about.
    if meta.get("dimensions_node") is not None:
        log.warning(
            "%s: declares both 'aspect_ratio_node' and 'dimensions_node'. These set the same "
            "thing; honouring the aspect ratio and ignoring 'size' for this workflow. Remove "
            "'dimensions_node' to silence this.",
            filename,
        )

    parsed = parse_aspect_ratio(inputs[field])
    if parsed is None:
        # The input exists, so injection still lands where it should — this only
        # costs the advertised default. Usually means the input is wired from
        # another node rather than holding a widget value, in which case injecting
        # replaces that link, so it is worth saying out loud.
        log.warning(
            "%s: aspect_ratio_node %r's %r is %r, not a ratio — the selector still works "
            "(an injected value replaces it) but no default is advertised.",
            filename,
            node_id,
            field,
            inputs[field],
        )
        return options, None
    return options, parsed.value


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
    aspect_ratio: str | None = None,
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

    # Aspect ratio. A workflow offering one exposes no pixel knob at all — the
    # megapixel budget and rounding stay baked into the node — so every ratio it
    # advertises renders within the VRAM the operator sized the graph for.
    if record.aspect_ratios:
        chosen = snap_aspect_ratio(aspect_ratio, record.aspect_ratios)
        node_id = str(meta["aspect_ratio_node"])
        node = workflow.get(node_id)
        if not isinstance(node, dict):
            # Missing, or present but not a node object. The scan validated this
            # id, but the graph is re-read per request while the meta is cached —
            # so an operator who renumbers or rewrites the node mid-process lands
            # here. Worth a warning rather than a silent no-op: the run still
            # succeeds, at the graph's own saved ratio, while
            # `effective_aspect_ratio` reports the snapped request. That
            # divergence is otherwise invisible.
            #
            # Guarding the type as well as the membership keeps a malformed graph
            # a warning rather than an AttributeError 500. Note it stops one level
            # short: a node that IS a dict whose `inputs` was rewritten to a
            # non-dict still raises on the assignment below. That's pre-existing
            # and matches the seed loop above, which indexes `node.get("inputs",
            # {})` the same way.
            log.warning(
                "Workflow %r: aspect_ratio_node %r is no longer a node in the graph; "
                "rendering at the graph's saved ratio and reporting the requested one. "
                "Restart to rescan, or set cache_workflows = false.",
                record.slug,
                node_id,
            )
        elif chosen is not None:
            field = meta.get("aspect_ratio_field", "aspect_ratio")
            node.setdefault("inputs", {})[field] = chosen.literal
    else:
        # Dimensions. Skipped entirely for a ratio workflow: the two would fight,
        # and the ratio node is downstream of nothing we could usefully set here.
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


def effective_aspect_ratio(record: WorkflowRecord, requested: str | None) -> str | None:
    """The canonical ratio a run will actually render at, for the response echo.

    Snapping means the value a client asked for is not always the value it gets,
    and a request naming none still renders at *something*. Reporting the real
    figure is what lets a client label or re-use a generation truthfully instead
    of recording its own request back.
    """
    chosen = snap_aspect_ratio(requested, record.aspect_ratios)
    return chosen.value if chosen is not None else record.aspect_ratio_default


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
