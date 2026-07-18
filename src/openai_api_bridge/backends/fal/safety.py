"""Deriving the loosest content-moderation settings for a fal model.

Mapping MODEL NAME -> knob needs a code change for every new model version
(a new Seedream, a new Nano Banana, ...). Instead we key on **field name** and
read the value out of the model's own OpenAPI input schema, so new versions of
a family are handled with no code change at all.

A sweep of fal's ~600 image models found the knobs collapse to just two field
names, both stable across versions within a family:

===========================  =====  ==========================================
field                        count  loosest value
===========================  =====  ==========================================
``enable_safety_checker``      132  ``False``
``safety_tolerance``            23  the highest value in the model's own enum
===========================  =====  ==========================================

Reading the enum rather than hardcoding matters: most models accept ``"1"``..
``"6"``, but ``fal-ai/flux-2-flex`` tops out at ``"5"`` — a hardcoded ``"6"``
is rejected. Defaults vary independently of range (``"4"`` for the Gemini
family, ``"2"`` for FLUX), so the default tells us nothing about the ceiling.

Two look-alike fields are deliberately **excluded**:

* ``safety_checker_version`` (``"v1"``/``"v2"``) selects *which* checker runs,
  not how strict it is — blindly picking the highest could make it stricter.
* ``has_nsfw_concepts`` is an *output* field (which images got flagged), not an
  input knob. We only ever read top-level properties of the request schema, so
  output fields can't leak in.

That exclusion is why this is an allowlist of known knobs rather than a regex
over anything safety-shaped.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

log = logging.getLogger(__name__)


# --- schema helpers ---------------------------------------------------------


def _resolve_ref(spec: dict[str, Any], node: Any) -> dict[str, Any] | None:
    """Follow a ``$ref`` into ``components.schemas``; pass other dicts through."""
    if not isinstance(node, dict):
        return None
    ref = node.get("$ref")
    if not isinstance(ref, str):
        return node
    # Only local refs ("#/components/schemas/Foo") appear in fal's specs.
    parts = ref.lstrip("#/").split("/")
    cur: Any = spec
    for p in parts:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(p)
    return cur if isinstance(cur, dict) else None


def input_schema(spec: dict[str, Any]) -> dict[str, Any] | None:
    """The request-body schema for a fal model's OpenAPI document.

    Prefers the POST ``requestBody`` whose ``$ref`` names an ``*Input`` schema
    (fal's convention), falling back to the first resolvable request body, then
    to any ``components.schemas`` entry ending in ``Input``.
    """
    fallback: dict[str, Any] | None = None
    paths = spec.get("paths")
    if isinstance(paths, dict):
        for item in paths.values():
            if not isinstance(item, dict):
                continue
            post = item.get("post")
            if not isinstance(post, dict):
                continue
            content = (post.get("requestBody") or {}).get("content")
            if not isinstance(content, dict):
                continue
            for media in content.values():
                if not isinstance(media, dict):
                    continue
                raw = media.get("schema")
                resolved = _resolve_ref(spec, raw)
                if not resolved or "properties" not in resolved:
                    continue
                ref = raw.get("$ref") if isinstance(raw, dict) else None
                if isinstance(ref, str) and ref.endswith("Input"):
                    return resolved
                fallback = fallback or resolved
    if fallback is not None:
        return fallback

    schemas = (spec.get("components") or {}).get("schemas")
    if isinstance(schemas, dict):
        for name, s in schemas.items():
            if name.endswith("Input") and isinstance(s, dict):
                return s
    return None


def _enum_of(schema: dict[str, Any]) -> list[Any] | None:
    if isinstance(schema.get("enum"), list) and schema["enum"]:
        return list(schema["enum"])
    # fal often wraps enums behind anyOf/allOf (e.g. for Optional[...]).
    for key in ("anyOf", "allOf", "oneOf"):
        for sub in schema.get(key) or []:
            if isinstance(sub, dict) and isinstance(sub.get("enum"), list) and sub["enum"]:
                return list(sub["enum"])
    return None


def _loosest_enum(schema: dict[str, Any]) -> Any | None:
    """Highest value in the field's own enum — fal's tolerance scales run
    strict->loose, so the maximum is the most permissive."""
    values = _enum_of(schema)
    if not values:
        return None
    try:
        return max(values, key=lambda v: float(v))
    except (TypeError, ValueError):
        # Non-numeric enum: last entry is the best guess at "loosest".
        return values[-1]


def _off(_schema: dict[str, Any]) -> Any:
    """A boolean checker we want disabled."""
    return False


# Allowlist of knobs we understand, keyed by input field name. Adding a new
# convention is a one-line change here — not a per-model entry.
_KNOBS: dict[str, Callable[[dict[str, Any]], Any]] = {
    "enable_safety_checker": _off,
    "safety_tolerance": _loosest_enum,
}


def safety_params_from_schema(spec: dict[str, Any]) -> dict[str, Any]:
    """Loosest-moderation body params derived from a model's OpenAPI schema.

    Returns ``{}`` when the model exposes no knob we recognise (e.g. fal's
    GPT-Image wrapper, which has no moderation field at all) — nothing is
    guessed, since an unknown field would 422 upstream.
    """
    schema = input_schema(spec)
    if not schema:
        return {}
    props = schema.get("properties")
    if not isinstance(props, dict):
        return {}
    out: dict[str, Any] = {}
    # Only top-level request properties are considered, so output-side fields
    # such as has_nsfw_concepts can never be picked up.
    for name, resolve in _KNOBS.items():
        prop = props.get(name)
        if not isinstance(prop, dict):
            continue
        value = resolve(prop)
        if value is not None:
            out[name] = value
    return out


# --- offline fallback -------------------------------------------------------

# Used only when schema introspection is disabled or the fetch fails. Matched
# as a substring of the model path, first match wins. Narrower and blunter than
# the schema path (it can't know a model's enum ceiling), but better than
# leaving guardrails at their defaults when fal's model API is unreachable.
_FALLBACK_RULES: tuple[tuple[str, dict[str, Any]], ...] = (
    ("seedream", {"enable_safety_checker": False}),
    ("nano-banana", {"safety_tolerance": "6"}),
    ("gemini", {"safety_tolerance": "6"}),
)


def fallback_safety_params(model_slug: str) -> dict[str, Any]:
    lowered = model_slug.lower()
    for needle, params in _FALLBACK_RULES:
        if needle in lowered:
            return dict(params)
    return {}
