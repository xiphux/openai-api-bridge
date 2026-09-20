"""Aspect-ratio parsing and nearest-match snapping.

A workflow node that takes an aspect ratio spells its options as *whole
literals* — ComfyUI's ``ResolutionSelector`` wants ``"16:9 (Widescreen)"``,
and a third-party node may offer ``"5:7 (Balanced Portrait)"`` that appears
in no standard list. Rather than enumerate a vocabulary the bridge would
have to chase, the meta declares the literals exactly as its node wants them
and this module derives the canonical ratio from each one's leading ``W:H``
token. A client sees ``{"value": "16:9", "label": "Widescreen"}``; the
literal never leaves the bridge.

That keeps a custom node's unusual ratios working with no code that knows
they exist, and lets a client draw an icon from the two numbers instead of
carrying a table of names.
"""

from __future__ import annotations

import logging
import math
import re
from dataclasses import dataclass
from typing import Any

log = logging.getLogger(__name__)

# The leading ratio token of a literal, plus whatever follows it. Tolerant of
# surrounding space because these strings are hand-written in meta files.
_RATIO = re.compile(r"^\s*(\d{1,6})\s*:\s*(\d{1,6})\s*(.*)$", re.DOTALL)


@dataclass(frozen=True, slots=True)
class AspectRatio:
    """One selectable aspect ratio.

    ``value`` is the ratio as the node spells it (``"16:9"``) and is what a
    client sends back. ``label`` is whatever followed that token — the
    parenthetical in ComfyUI's own spellings, but see
    :func:`parse_aspect_ratio` for the unparenthesised case — and is ``None``
    for a bare ratio. ``literal`` is the exact string the upstream node
    expects — bridge-internal.

    ``ratio`` is ``width / height``, precomputed because snapping compares it
    for every option on every request.
    """

    value: str
    label: str | None
    literal: str
    ratio: float


def parse_aspect_ratio(literal: object) -> AspectRatio | None:
    """Derive an :class:`AspectRatio` from one declared literal.

    Returns ``None`` for anything that doesn't start with a ``W:H`` token, or
    whose height is zero — a meta typo shouldn't take the workflow down, so
    the caller drops the entry and warns.
    """
    if not isinstance(literal, str):
        return None
    m = _RATIO.match(literal)
    if m is None:
        return None
    w, h = int(m.group(1)), int(m.group(2))
    if w <= 0 or h <= 0:
        return None
    # Deliberately NOT reduced. Aspect ratios are conventional names, not
    # fractions: reducing turns the universally-recognised "21:9" into "7:3"
    # (and true ultrawide is 64:27 regardless — 21:9 is a marketing name that
    # survived). The node's own spelling is the canonical one, so only the
    # spacing is normalised. Equality between differently-spelled ratios is
    # snapping's job, and it compares the float, not this string.
    value = f"{w}:{h}"
    # "16:9 (Widescreen)" → "Widescreen"; "16:9" → None. A trailing remainder
    # that isn't parenthesised is taken as-is rather than discarded, since a
    # node naming its options "16:9 widescreen" is no less a label.
    rest = m.group(3).strip()
    if rest.startswith("(") and rest.endswith(")"):
        rest = rest[1:-1].strip()
    return AspectRatio(value=value, label=rest or None, literal=literal, ratio=w / h)


def parse_aspect_ratios(raw: object, *, where: str) -> tuple[AspectRatio, ...]:
    """Parse a meta ``aspect_ratios`` declaration into options, in declared order.

    Declared order is preserved and load-bearing: it's the order a client
    renders, and it breaks ties when two options are equally near a requested
    ratio. Unparseable entries and duplicate canonical values are dropped with
    a warning naming ``where`` (the workflow file), so an operator finds out
    from the log rather than from a menu that's quietly missing a row.
    """
    if not isinstance(raw, list):
        log.warning("%s: 'aspect_ratios' must be a list of strings; ignoring", where)
        return ()
    out: list[AspectRatio] = []
    seen: set[str] = set()
    for entry in raw:
        parsed = parse_aspect_ratio(entry)
        if parsed is None:
            log.warning(
                "%s: aspect ratio %r does not start with a 'W:H' token; dropping", where, entry
            )
            continue
        if parsed.value in seen:
            log.warning(
                "%s: aspect ratio %r duplicates %r already declared; dropping",
                where,
                entry,
                parsed.value,
            )
            continue
        seen.add(parsed.value)
        out.append(parsed)
    return tuple(out)


def snap_aspect_ratio(requested: object, options: tuple[AspectRatio, ...]) -> AspectRatio | None:
    """Resolve a requested ratio to the nearest option this model accepts.

    Nearest rather than rejected, following the same reasoning as fal's
    ``duration`` snapping: a client holding one menu (a remembered preference,
    or a fan-out across several models) shouldn't get a 400 because this
    model's menu is spelled differently. Falling back to the *workflow's*
    default instead would be worse than either — asking for ultrawide and
    receiving the workflow's baked-in portrait is further from the request
    than any option in the list.

    Distance is measured in log space, so 2:1 sits as far from 1:1 as 1:2
    does. An exact canonical match short-circuits, which keeps the common case
    free of float comparison and honours declared order.

    Returns ``None`` when there are no options or the request is unparseable —
    the caller then injects nothing and the workflow's own default stands.
    """
    if not options:
        return None
    want = parse_aspect_ratio(requested)
    if want is None:
        if requested is not None:
            log.debug("Unparseable aspect_ratio %r; leaving the workflow default", requested)
        return None
    for option in options:
        if option.value == want.value:
            return option
    target = math.log(want.ratio)
    nearest = min(options, key=lambda o: abs(math.log(o.ratio) - target))
    log.debug(
        "Aspect ratio %r is not offered; snapping to nearest %r",
        want.value,
        nearest.value,
    )
    return nearest


def advertised(options: tuple[AspectRatio, ...]) -> list[dict[str, Any]]:
    """Render options for a ``/v1/models`` row.

    ``label`` is omitted rather than null when the literal carried none,
    matching the rest of the model-metadata extensions.
    """
    return [
        {"value": o.value} if o.label is None else {"value": o.value, "label": o.label}
        for o in options
    ]
