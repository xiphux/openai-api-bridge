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

from .schema import enum_of, input_properties

log = logging.getLogger(__name__)


def _loosest_enum(schema: dict[str, Any], spec: dict[str, Any] | None = None) -> Any | None:
    """Highest value in the field's own enum — fal's tolerance scales run
    strict->loose, so the maximum is the most permissive."""
    values = enum_of(schema, spec)
    if not values:
        return None
    try:
        return max(values, key=lambda v: float(v))
    except (TypeError, ValueError):
        # Non-numeric enum: last entry is the best guess at "loosest".
        return values[-1]


def _off(_schema: dict[str, Any], _spec: dict[str, Any] | None = None) -> Any:
    """A boolean checker we want disabled."""
    return False


# Allowlist of knobs we understand, keyed by input field name. Adding a new
# convention is a one-line change here — not a per-model entry.
_KNOBS: dict[str, Callable[[dict[str, Any], dict[str, Any] | None], Any]] = {
    "enable_safety_checker": _off,
    "safety_tolerance": _loosest_enum,
}


def safety_params_from_schema(spec: dict[str, Any]) -> dict[str, Any]:
    """Loosest-moderation body params derived from a model's OpenAPI schema.

    Returns ``{}`` when the model exposes no knob we recognise (e.g. fal's
    GPT-Image wrapper, which has no moderation field at all) — nothing is
    guessed, since an unknown field would 422 upstream.
    """
    props = input_properties(spec)
    out: dict[str, Any] = {}
    # Only top-level request properties are considered, so output-side fields
    # such as has_nsfw_concepts can never be picked up.
    for name, resolve in _KNOBS.items():
        prop = props.get(name)
        if not isinstance(prop, dict):
            continue
        value = resolve(prop, spec)
        if value is not None:
            out[name] = value
    return out


# --- offline fallback -------------------------------------------------------

# Used for a model whenever schema-derived params aren't available for it:
# introspection disabled, the lookup failed, or the lookup succeeded but that
# particular model was absent from the response (its siblings can still get
# schema-derived params in the same run). Matched as a substring of the model
# path, first match wins. Narrower and blunter than the schema path — it can't
# know a model's enum ceiling — but better than leaving guardrails at their
# defaults.
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
