"""fal OpenAPI schema reading: pure functions, pinned table by table.

These decide which request fields reach fal — the moderation knob and the
clip-length spelling each model accepts — from shapes fal's live specs use.
A wrong answer here is silent: the field is omitted or misspelled, and the
model either ignores it or 422s. See the module docstring for why they read
the model's own schema instead of a static table.
"""

from __future__ import annotations

from typing import Any

import pytest

from openai_api_bridge.backends.fal.schema import (
    duration_params,
    duration_property,
    enum_of,
    input_properties,
    input_schema,
    resolve_ref,
)


def _spec(
    body_ref: str | None = "#/components/schemas/ModelInput",
    schemas: dict[str, Any] | None = None,
    inline_body: dict[str, Any] | None = None,
) -> dict[str, Any]:
    schema: dict[str, Any] = {"$ref": body_ref} if body_ref else (inline_body or {})
    return {
        "paths": {
            "/model": {
                "post": {"requestBody": {"content": {"application/json": {"schema": schema}}}}
            }
        },
        "components": {"schemas": schemas or {}},
    }


# --- resolve_ref ---------------------------------------------------------------


def test_resolve_ref_follows_a_local_ref() -> None:
    spec = {"components": {"schemas": {"A": {"type": "object"}}}}
    assert resolve_ref(spec, {"$ref": "#/components/schemas/A"}) == {"type": "object"}


@pytest.mark.parametrize(
    ("node", "expected"),
    [
        ({"type": "string"}, {"type": "string"}),  # not a ref: passed through
        ("not a dict", None),
        ({"$ref": "#/components/schemas/Missing"}, None),
        ({"$ref": "#/components/schemas/A/type"}, None),  # lands on a non-dict
        ({"$ref": "#/components/nope/deeper"}, None),  # walks through a non-dict
    ],
)
def test_resolve_ref_edge_cases(node: Any, expected: Any) -> None:
    spec = {"components": {"schemas": {"A": {"type": "object"}}, "nope": "leaf"}}
    assert resolve_ref(spec, node) == expected


# --- input_schema / input_properties -----------------------------------------------


def test_input_schema_prefers_the_body_ref_named_input() -> None:
    wanted = {"properties": {"prompt": {}}}
    spec = _spec(schemas={"ModelInput": wanted})
    assert input_schema(spec) is wanted


def test_input_schema_takes_an_inline_body_when_no_input_ref() -> None:
    inline = {"properties": {"prompt": {}}}
    assert input_schema(_spec(body_ref=None, inline_body=inline)) == inline


def test_input_schema_skips_bodies_without_properties() -> None:
    spec = _spec(body_ref=None, inline_body={"type": "object"})
    spec["components"]["schemas"] = {"ThingInput": {"properties": {"seed": {}}}}
    # No usable request body → the components fallback, by name.
    assert input_schema(spec) == {"properties": {"seed": {}}}


@pytest.mark.parametrize(
    "spec",
    [
        {},
        {"paths": "not a dict"},
        {"paths": {"/x": "not a dict"}},
        {"paths": {"/x": {"get": {}}}},  # no POST
        {"paths": {"/x": {"post": {"requestBody": {"content": "nope"}}}}},
        {"paths": {"/x": {"post": {"requestBody": {"content": {"a/b": "nope"}}}}}},
        {"components": {"schemas": {"ModelOutput": {"properties": {}}}}},  # output only
    ],
    ids=["empty", "paths-str", "item-str", "no-post", "content-str", "media-str", "only-output"],
)
def test_input_schema_is_none_when_nothing_qualifies(spec: dict[str, Any]) -> None:
    assert input_schema(spec) is None
    assert input_properties(spec) == {}


def test_input_properties_tolerates_a_schema_without_a_properties_map() -> None:
    spec = {"components": {"schemas": {"ModelInput": {"properties": ["not", "a", "map"]}}}}
    assert input_properties(spec) == {}


# --- enum_of -----------------------------------------------------------------------


REFD = {"components": {"schemas": {"Durations": {"enum": ["5", "10"]}}}}


@pytest.mark.parametrize(
    ("schema", "spec", "expected"),
    [
        ({"enum": ["1", "2"]}, None, ["1", "2"]),
        ({"anyOf": [{"type": "null"}, {"enum": ["4s", "8s"]}]}, None, ["4s", "8s"]),
        ({"allOf": ["junk", {"enum": [1, 2]}]}, None, [1, 2]),
        ({"anyOf": [{"$ref": "#/components/schemas/Durations"}]}, REFD, ["5", "10"]),
        ({"$ref": "#/components/schemas/Durations"}, REFD, ["5", "10"]),
        # Without the spec a ref can't be followed: "no values", not a crash.
        ({"$ref": "#/components/schemas/Durations"}, None, None),
        ({"enum": []}, None, None),
        ({"type": "integer"}, REFD, None),
    ],
)
def test_enum_of(schema: dict[str, Any], spec: dict[str, Any] | None, expected: Any) -> None:
    assert enum_of(schema, spec) == expected


# --- duration_property / duration_params ---------------------------------------------


def test_duration_property_inlines_a_ref_encoded_enum() -> None:
    """The adapter caches the property alone, so a ref must be resolved now."""
    spec = _spec(
        schemas={
            "ModelInput": {
                "properties": {"duration": {"anyOf": [{"$ref": "#/components/schemas/D"}]}}
            },
            "D": {"enum": ["6", "10"]},
        }
    )
    prop = duration_property(spec)
    assert prop is not None
    assert prop["enum"] == ["6", "10"]
    assert duration_params(prop, 7) == {"duration": "6"}


def test_duration_property_is_none_without_a_duration_input() -> None:
    spec = _spec(schemas={"ModelInput": {"properties": {"num_frames": {"type": "integer"}}}})
    assert duration_property(spec) is None
    assert duration_params(None, 5) == {}


@pytest.mark.parametrize(
    ("prop", "seconds", "expected"),
    [
        # Enums: the closest accepted value, in the model's own spelling.
        ({"enum": ["4s", "6s", "8s"]}, 7.9, {"duration": "8s"}),  # veo3
        ({"enum": ["4s", "6s", "8s"]}, 5, {"duration": "6s"}),  # a tie goes longer
        ({"enum": ["5", "10"]}, 6, {"duration": "5"}),  # Kling
        ({"enum": [5, 10]}, 9, {"duration": 10}),  # numeric enum
        ({"enum": ["short", "long"]}, 5, {}),  # nothing numeric to match
        ({"enum": [True, "7s"]}, 1, {"duration": "7s"}),  # a bool is not a number
        # Free-form: follow the declared type.
        ({"type": "integer"}, 4.6, {"duration": 5}),
        ({"type": "number"}, 4.5, {"duration": 4.5}),
        ({"type": "string"}, 4.4, {"duration": "4"}),
        ({"description": "no type, no enum"}, 5, {}),
    ],
)
def test_duration_params(prop: dict[str, Any], seconds: float, expected: dict[str, Any]) -> None:
    assert duration_params(prop, seconds) == expected
