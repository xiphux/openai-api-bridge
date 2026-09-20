"""Aspect-ratio parsing, snapping, and the meta declaration that drives them."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest

from openai_api_bridge.backends.comfyui.workflows import (
    effective_aspect_ratio,
    prepare_workflow,
    read_graph_text,
    scan_workflows,
)
from openai_api_bridge.util.aspect import (
    advertised,
    parse_aspect_ratio,
    parse_aspect_ratios,
    snap_aspect_ratio,
)

# The stock ComfyUI ResolutionSelector menu, verbatim.
STOCK = [
    "1:1 (Square)",
    "2:3 (Portrait Photo)",
    "3:2 (Photo)",
    "3:4 (Portrait Standard)",
    "4:3 (Standard)",
    "9:16 (Portrait Widescreen)",
    "16:9 (Widescreen)",
    "21:9 (Ultrawide)",
]


# --- parsing ---------------------------------------------------------------


def test_a_literal_splits_into_ratio_and_label() -> None:
    ar = parse_aspect_ratio("16:9 (Widescreen)")
    assert ar is not None
    assert (ar.value, ar.label, ar.literal) == ("16:9", "Widescreen", "16:9 (Widescreen)")
    assert ar.ratio == pytest.approx(16 / 9)


def test_a_bare_ratio_has_no_label() -> None:
    ar = parse_aspect_ratio("4:3")
    assert ar is not None
    assert (ar.value, ar.label) == ("4:3", None)


def test_a_custom_nodes_unusual_ratio_parses_like_any_other() -> None:
    """The point of deriving from the literal: nothing enumerates the options."""
    ar = parse_aspect_ratio("5:7 (Balanced Portrait)")
    assert ar is not None
    assert (ar.value, ar.label) == ("5:7", "Balanced Portrait")


def test_conventional_ratios_are_not_reduced() -> None:
    """21:9 is a name, not a fraction — reducing it to 7:3 would rename it into
    something no client or user would recognise."""
    ar = parse_aspect_ratio("21:9 (Ultrawide)")
    assert ar is not None
    assert ar.value == "21:9"


@pytest.mark.parametrize(
    "literal",
    ["", "Square", "16-9", ":", "16:", "16:0 (Degenerate)", "0:9", None, 42, ["16:9"]],
)
def test_junk_does_not_parse(literal: object) -> None:
    assert parse_aspect_ratio(literal) is None


def test_whitespace_around_the_token_is_normalised() -> None:
    ar = parse_aspect_ratio("  16 : 9  (Widescreen)  ")
    assert ar is not None
    assert (ar.value, ar.label) == ("16:9", "Widescreen")


def test_declared_order_is_preserved() -> None:
    options = parse_aspect_ratios(STOCK, where="t.json")
    assert [o.value for o in options] == [
        "1:1",
        "2:3",
        "3:2",
        "3:4",
        "4:3",
        "9:16",
        "16:9",
        "21:9",
    ]


def test_unparseable_entries_are_dropped_with_a_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        options = parse_aspect_ratios(["16:9 (Widescreen)", "Nonsense"], where="t.json")
    assert [o.value for o in options] == ["16:9"]
    assert "t.json" in caplog.text
    assert "Nonsense" in caplog.text


def test_duplicates_are_dropped(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        options = parse_aspect_ratios(["16:9 (A)", "16:9 (B)"], where="t.json")
    assert [o.literal for o in options] == ["16:9 (A)"]
    assert "duplicates" in caplog.text


def test_a_non_list_declaration_is_rejected(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        assert parse_aspect_ratios("16:9", where="t.json") == ()
    assert "must be a list" in caplog.text


def test_a_label_is_omitted_rather_than_nulled_when_advertised() -> None:
    assert advertised(parse_aspect_ratios(["16:9 (Widescreen)", "4:3"], where="t")) == [
        {"value": "16:9", "label": "Widescreen"},
        {"value": "4:3"},
    ]


# --- snapping --------------------------------------------------------------


def test_an_exact_match_is_returned_as_declared() -> None:
    chosen = snap_aspect_ratio("16:9", parse_aspect_ratios(STOCK, where="t"))
    assert chosen is not None
    assert chosen.literal == "16:9 (Widescreen)"


def test_a_differently_spelled_equal_ratio_still_matches() -> None:
    """A client sending the reduced form of a conventional name must not miss."""
    chosen = snap_aspect_ratio("7:3", parse_aspect_ratios(STOCK, where="t"))
    assert chosen is not None
    assert chosen.value == "21:9"


@pytest.mark.parametrize(
    ("requested", "expected"),
    [
        # Ultrawide against a menu topping out at 16:9 → the widest landscape,
        # NOT the workflow's baked-in portrait default.
        ("21:9", "16:9"),
        # Portrait widescreen against a portrait-and-square menu.
        ("9:16", "5:7"),
        ("1:1", "1:1"),
    ],
)
def test_a_missing_ratio_snaps_to_the_nearest_offered(requested: str, expected: str) -> None:
    options = parse_aspect_ratios(
        ["1:1 (Square)", "5:7 (Balanced Portrait)", "16:9 (Widescreen)"], where="t"
    )
    chosen = snap_aspect_ratio(requested, options)
    assert chosen is not None
    assert chosen.value == expected


def test_distance_is_symmetric_in_log_space() -> None:
    """2:1 must sit as far from 1:1 as 1:2 does, so a tie stays a tie and falls
    to declared order rather than to whichever side arithmetic favours."""
    options = parse_aspect_ratios(["2:1 (Wide)", "1:2 (Tall)"], where="t")
    chosen = snap_aspect_ratio("1:1", options)
    assert chosen is not None
    assert chosen.value == "2:1"  # first declared wins the tie


def test_snapping_is_deterministic_across_calls() -> None:
    options = parse_aspect_ratios(["2:1 (Wide)", "1:2 (Tall)"], where="t")
    picks = {snap_aspect_ratio("1:1", options) for _ in range(20)}
    assert len(picks) == 1


@pytest.mark.parametrize("requested", [None, "", "garbage", 42])
def test_an_unusable_request_selects_nothing(requested: object) -> None:
    """So the caller injects nothing and the workflow's own default stands."""
    assert snap_aspect_ratio(requested, parse_aspect_ratios(STOCK, where="t")) is None


def test_no_options_selects_nothing() -> None:
    assert snap_aspect_ratio("16:9", ()) is None


# --- the meta declaration --------------------------------------------------


def _write(tmp_path: Path, meta: dict[str, Any], graph: dict[str, Any]) -> None:
    (tmp_path / "W.json").write_text(json.dumps(graph))
    (tmp_path / "W.meta.json").write_text(json.dumps(meta))


def _ratio_graph(current: str = "16:9 (Widescreen)", field: str = "aspect_ratio") -> dict[str, Any]:
    return {
        "1": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}},
        "5": {
            "class_type": "ResolutionSelector",
            "inputs": {field: current, "megapixels": 0.5, "multiple": 32},
        },
    }


def test_a_declared_node_and_list_are_resolved_at_scan_time(tmp_path: Path) -> None:
    _write(
        tmp_path,
        {"positive_prompt_node": "1", "aspect_ratio_node": "5", "aspect_ratios": STOCK},
        _ratio_graph(),
    )
    record = scan_workflows(tmp_path)["w"]
    assert [o.value for o in record.aspect_ratios] == [
        "1:1",
        "2:3",
        "3:2",
        "3:4",
        "4:3",
        "9:16",
        "16:9",
        "21:9",
    ]


def test_the_default_is_read_from_the_graphs_saved_value(tmp_path: Path) -> None:
    """Costs no meta field, and means the advertised default can't drift from
    what the workflow actually does."""
    _write(
        tmp_path,
        {"positive_prompt_node": "1", "aspect_ratio_node": "5", "aspect_ratios": STOCK},
        _ratio_graph(current="2:3 (Portrait Photo)"),
    )
    assert scan_workflows(tmp_path)["w"].aspect_ratio_default == "2:3"


def test_a_custom_field_name_is_honoured(tmp_path: Path) -> None:
    """FluxResolutionNode spells its neighbours differently from
    ResolutionSelector, so the field name can't be assumed."""
    _write(
        tmp_path,
        {
            "positive_prompt_node": "1",
            "aspect_ratio_node": "5",
            "aspect_ratio_field": "ratio",
            "aspect_ratios": STOCK,
        },
        _ratio_graph(current="4:3 (Standard)", field="ratio"),
    )
    assert scan_workflows(tmp_path)["w"].aspect_ratio_default == "4:3"


@pytest.mark.parametrize(
    "meta_extra",
    [
        {"aspect_ratio_node": "5"},  # node without a list
        {"aspect_ratios": STOCK},  # list without a node
    ],
)
def test_half_a_declaration_disables_the_feature(
    tmp_path: Path, meta_extra: dict[str, Any], caplog: pytest.LogCaptureFixture
) -> None:
    _write(tmp_path, {"positive_prompt_node": "1", **meta_extra}, _ratio_graph())
    with caplog.at_level(logging.WARNING):
        record = scan_workflows(tmp_path)["w"]
    assert record.aspect_ratios == ()
    assert "is missing" in caplog.text


def test_a_stale_node_id_disables_the_feature(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """ComfyUI ids aren't stable across a re-export, so this is the likely
    breakage and it is otherwise silent."""
    _write(
        tmp_path,
        {"positive_prompt_node": "1", "aspect_ratio_node": "99", "aspect_ratios": STOCK},
        _ratio_graph(),
    )
    with caplog.at_level(logging.WARNING):
        record = scan_workflows(tmp_path)["w"]
    assert record.aspect_ratios == ()
    assert "not in the graph" in caplog.text


def test_declaring_both_knobs_warns_and_the_ratio_wins(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    graph = _ratio_graph()
    graph["9"] = {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512}}
    _write(
        tmp_path,
        {
            "positive_prompt_node": "1",
            "aspect_ratio_node": "5",
            "aspect_ratios": STOCK,
            "dimensions_node": "9",
        },
        graph,
    )
    with caplog.at_level(logging.WARNING):
        record = scan_workflows(tmp_path)["w"]
    assert "same thing" in caplog.text

    # And `size` is ignored for it: the two would fight, and the ratio node
    # computes the dimensions the graph actually consumes.
    workflow = prepare_workflow(
        record,
        read_graph_text(record),
        prompt_text="x",
        width=4096,
        height=4096,
        aspect_ratio="1:1",
    )
    assert workflow["9"]["inputs"]["width"] == 512
    assert workflow["5"]["inputs"]["aspect_ratio"] == "1:1 (Square)"


def test_a_workflow_without_the_declaration_still_honours_size(tmp_path: Path) -> None:
    graph = {
        "1": {"class_type": "CLIPTextEncode", "inputs": {"text": ""}},
        "9": {"class_type": "EmptyLatentImage", "inputs": {"width": 512, "height": 512}},
    }
    _write(tmp_path, {"positive_prompt_node": "1", "dimensions_node": "9"}, graph)
    record = scan_workflows(tmp_path)["w"]
    assert record.aspect_ratios == ()
    workflow = prepare_workflow(
        record, read_graph_text(record), prompt_text="x", width=1024, height=768
    )
    assert (workflow["9"]["inputs"]["width"], workflow["9"]["inputs"]["height"]) == (1024, 768)


# --- injection + echo ------------------------------------------------------


@pytest.fixture
def ratio_record(tmp_path: Path) -> Any:
    _write(
        tmp_path,
        {"positive_prompt_node": "1", "aspect_ratio_node": "5", "aspect_ratios": STOCK},
        _ratio_graph(),
    )
    return scan_workflows(tmp_path)["w"]


def test_the_nodes_own_literal_is_injected_not_the_canonical_value(ratio_record: Any) -> None:
    """The node is a combo widget: it accepts its own spelling and nothing else."""
    workflow = prepare_workflow(
        ratio_record, read_graph_text(ratio_record), prompt_text="x", aspect_ratio="21:9"
    )
    assert workflow["5"]["inputs"]["aspect_ratio"] == "21:9 (Ultrawide)"


def test_the_megapixel_budget_is_left_alone(ratio_record: Any) -> None:
    """The whole safety argument: a client can pick any advertised ratio without
    being able to ask for an image large enough to OOM the box."""
    workflow = prepare_workflow(
        ratio_record, read_graph_text(ratio_record), prompt_text="x", aspect_ratio="21:9"
    )
    assert workflow["5"]["inputs"]["megapixels"] == 0.5
    assert workflow["5"]["inputs"]["multiple"] == 32


def test_naming_no_ratio_leaves_the_graph_untouched(ratio_record: Any) -> None:
    workflow = prepare_workflow(ratio_record, read_graph_text(ratio_record), prompt_text="x")
    assert workflow["5"]["inputs"]["aspect_ratio"] == "16:9 (Widescreen)"


def test_the_echo_reports_the_snapped_value_not_the_request(ratio_record: Any) -> None:
    assert effective_aspect_ratio(ratio_record, "5:7") == "3:4"


def test_the_echo_falls_back_to_the_workflow_default(ratio_record: Any) -> None:
    assert effective_aspect_ratio(ratio_record, None) == "16:9"


def test_the_echo_is_none_for_a_workflow_with_no_ratios(tmp_path: Path) -> None:
    _write(tmp_path, {"positive_prompt_node": "1"}, _ratio_graph())
    record = scan_workflows(tmp_path)["w"]
    assert effective_aspect_ratio(record, "16:9") is None
