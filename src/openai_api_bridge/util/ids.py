"""ID/slug helpers."""

from __future__ import annotations

import re

_NON_SLUG_RE = re.compile(r"[^a-z0-9-]+")
_HYPHEN_RUN_RE = re.compile(r"-+")
_SEPARATOR_RE = re.compile(r"[\s/:.\\]+")


def slugify(name: str) -> str:
    """Lowercase + hyphenate. Collapses runs of hyphens. Strips leading/trailing.

    Examples:
        ``"LTX-V T2V (Fast)"`` → ``"ltx-v-t2v-fast"``
        ``"Flux.2 Klein"``    → ``"flux-2-klein"``
    """
    s = name.lower().strip()
    s = _SEPARATOR_RE.sub("-", s)
    s = _NON_SLUG_RE.sub("", s)
    s = _HYPHEN_RUN_RE.sub("-", s)
    return s.strip("-")
