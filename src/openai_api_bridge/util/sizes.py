"""Helpers for parsing OpenAI-shaped dimension strings."""

from __future__ import annotations


def parse_size(size: str | None) -> tuple[int, int]:
    """Parse OpenAI's ``"WIDTHxHEIGHT"`` size string. Returns (0, 0) for falsy/invalid input.

    The convention "0 means use the upstream default" is preserved so callers
    can apply per-provider defaults via ``or``-fallthrough.

    A non-positive dimension counts as invalid, and invalidates the pair rather
    than just its own half. Both halves matter:

    * ``int()`` happily parses ``"-1"``, so ``"-1x100"`` used to return
      ``(-1, 100)``. Callers that guard with ``> 0`` (fal, ComfyUI) dropped it;
      Venice's per-dimension ``w or default`` did not, because ``-1`` is truthy
      — so a negative width reached the upstream as a real request parameter.
    * Invalidating the pair keeps the docstring's promise literally true and
      avoids half-applying nonsense: ``"1024x0"`` now falls back to the
      provider's defaults for both, rather than honouring a width that arrived
      alongside a height the caller clearly didn't mean.

    Note this only makes malformed input fall back to defaults *reliably*. It
    is deliberately not validation — nothing here rejects a request, so
    ``"garbage"``, ``"1024X1024"`` (capital X) and OpenAI's ``"auto"`` all
    still mean "use the default", silently, as they always have.
    """
    if not size or "x" not in size:
        return 0, 0
    try:
        w_str, _, h_str = size.partition("x")
        width, height = int(w_str), int(h_str)
    except ValueError:
        return 0, 0
    if width <= 0 or height <= 0:
        return 0, 0
    return width, height
