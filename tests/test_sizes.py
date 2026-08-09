"""``parse_size``: what reaches a provider as a dimension, and what doesn't.

The (0, 0) return is a sentinel meaning "no size given" — every backend turns
it into its own default. So the property that matters here isn't parsing, it's
that nothing a client can send arrives downstream as a *number* the caller
didn't mean.
"""

from __future__ import annotations

import pytest

from openai_api_bridge.util.sizes import parse_size


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1024x1024", (1024, 1024)),
        ("512x768", (512, 768)),
        ("1x1", (1, 1)),
    ],
)
def test_wellformed_sizes_parse(raw: str, expected: tuple[int, int]) -> None:
    assert parse_size(raw) == expected


@pytest.mark.parametrize(
    "raw",
    [
        None,
        "",
        "garbage",
        "axb",
        "1024x",
        "x1024",
        "1024",
        "1024X1024",  # capital X is not the separator
        "auto",  # OpenAI's own vocabulary for gpt-image-1
        "1024x1024x1024",
    ],
)
def test_unparseable_sizes_mean_no_size(raw: str | None) -> None:
    """Unchanged behaviour, pinned deliberately: these fall back to the
    provider default rather than being rejected. `parse_size` is not
    validation and this test is not an argument that it shouldn't be — it
    records that today it isn't."""
    assert parse_size(raw) == (0, 0)


@pytest.mark.parametrize("raw", ["-1x100", "100x-1", "-5x-5", "0x0", "1024x0", "0x1024"])
def test_non_positive_dimensions_never_reach_a_provider(raw: str) -> None:
    """`int()` parses "-1" perfectly well, so these used to survive as real
    numbers. fal and ComfyUI guard with `> 0`, but Venice applies its default
    with `w or default` — and `-1` is truthy, so a negative width went
    upstream as a request parameter. Invalidating the whole pair also stops
    "1024x0" half-applying a width the caller didn't mean."""
    assert parse_size(raw) == (0, 0)
