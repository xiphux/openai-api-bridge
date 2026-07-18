"""Reading request parameters out of a fal model's OpenAPI schema.

fal exposes each model's native input schema, and the bridge leans on that
rather than hardcoding per-model knowledge — models change, and their accepted
values differ in ways no static table survives:

* moderation is ``enable_safety_checker`` (bool) on some families and
  ``safety_tolerance`` (enum) on others, and the enum's ceiling varies —
  most accept ``"1"``..``"6"``, ``fal-ai/flux-2-flex`` stops at ``"5"``.
* ``duration`` is an enum whose *literal spelling* differs per model:
  ``fal-ai/veo3`` wants ``"4s"``/``"6s"``/``"8s"``, Kling wants ``"5"``/``"10"``,
  Hailuo ``"6"``/``"10"``, and ``wan`` has no duration field at all (it counts
  frames). Sending a plain number is rejected by most of them.

So the rule throughout: find the field by *name*, then take its accepted values
from the model's own schema. See :mod:`.safety` for the moderation side.
"""

from __future__ import annotations

from typing import Any


def resolve_ref(spec: dict[str, Any], node: Any) -> dict[str, Any] | None:
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
                resolved = resolve_ref(spec, raw)
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


def input_properties(spec: dict[str, Any]) -> dict[str, Any]:
    """Top-level request properties, or ``{}``.

    Only top-level *request* properties — never the response side — so output
    fields (``has_nsfw_concepts``, say) can't be mistaken for inputs.
    """
    schema = input_schema(spec)
    if not schema:
        return {}
    props = schema.get("properties")
    return props if isinstance(props, dict) else {}


def enum_of(schema: dict[str, Any], spec: dict[str, Any] | None = None) -> list[Any] | None:
    """A property's accepted values, unwrapping fal's anyOf/allOf nesting.

    ``spec`` lets a ``$ref``-encoded enum be followed. fal inlines these today
    — every knob checked against the live API has its values in place — but the
    failure mode if that changes is silent: an unresolved enum reads as "no
    values", and the moderation knob is simply omitted with nothing to see in
    the request. Cheap insurance for the one thing this backend exists to set.
    """
    if isinstance(schema.get("enum"), list) and schema["enum"]:
        return list(schema["enum"])
    for key in ("anyOf", "allOf", "oneOf"):
        for sub in schema.get(key) or []:
            if not isinstance(sub, dict):
                continue
            if isinstance(sub.get("enum"), list) and sub["enum"]:
                return list(sub["enum"])
            if spec is not None and "$ref" in sub:
                target = resolve_ref(spec, sub)
                if target and isinstance(target.get("enum"), list) and target["enum"]:
                    return list(target["enum"])
    if spec is not None and "$ref" in schema:
        target = resolve_ref(spec, schema)
        if target and isinstance(target.get("enum"), list) and target["enum"]:
            return list(target["enum"])
    return None


def _as_number(value: Any) -> float | None:
    """Numeric value of an enum entry, tolerating unit suffixes like ``"8s"``."""
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if not isinstance(value, str):
        return None
    digits = "".join(c for c in value if c.isdigit() or c == ".")
    try:
        return float(digits) if digits else None
    except ValueError:
        return None


def duration_property(spec: dict[str, Any]) -> dict[str, Any] | None:
    """The model's ``duration`` input property, if it has one.

    Any ``$ref``-encoded enum is resolved and inlined here, while the whole
    document is still in hand — the adapter caches this property alone, not the
    spec, so a reference left dangling would be unresolvable by the time a
    request needs it.
    """
    prop = input_properties(spec).get("duration")
    if not isinstance(prop, dict):
        return None
    if not prop.get("enum"):
        values = enum_of(prop, spec)
        if values:
            return {**prop, "enum": values}
    return prop


def duration_params(prop: dict[str, Any] | None, seconds: float) -> dict[str, Any]:
    """Body fields expressing a requested clip length for this model.

    Returns ``{}`` when the model has no ``duration`` input — ``wan`` counts
    frames instead, and forcing a field it doesn't accept would 422.

    When ``duration`` is an enum we return the **closest accepted value in the
    model's own spelling**, so a request for 5s becomes ``"6s"`` on veo3
    (``4s``/``6s``/``8s``) but ``"5"`` on Kling (``5``/``10``). Picking the
    nearest rather than rejecting keeps a generic client working across models
    whose menus don't line up; the response is honest about what was produced
    because the asset itself is whatever the model made.
    """
    if prop is None:
        return {}
    values = enum_of(prop)
    if not values:
        # Free-form numeric duration: pass the request through as-is, matching
        # the declared type where we can tell.
        declared = prop.get("type")
        if declared == "integer":
            return {"duration": round(seconds)}
        if declared == "number":
            return {"duration": float(seconds)}
        if declared == "string":
            return {"duration": str(round(seconds))}
        return {}
    numbered = [(v, _as_number(v)) for v in values]
    usable = [(v, n) for v, n in numbered if n is not None]
    if not usable:
        return {}
    # Closest accepted value; ties go to the longer one. A request for 5s
    # against veo3's 4s/6s/8s is exactly such a tie, and overshooting is the
    # kinder miss — silently returning a clip shorter than asked for loses
    # content the caller wanted.
    closest = min(usable, key=lambda pair: (abs(pair[1] - seconds), -pair[1]))[0]
    return {"duration": closest}
