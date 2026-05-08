"""Helpers for parsing OpenAI-shaped dimension strings."""

from __future__ import annotations


def parse_size(size: str | None) -> tuple[int, int]:
    """Parse OpenAI's ``"WIDTHxHEIGHT"`` size string. Returns (0, 0) for falsy/invalid input.

    The convention "0 means use the upstream default" is preserved so callers
    can apply per-provider defaults via ``or``-fallthrough.
    """
    if not size or "x" not in size:
        return 0, 0
    try:
        w_str, _, h_str = size.partition("x")
        return int(w_str), int(h_str)
    except ValueError:
        return 0, 0
