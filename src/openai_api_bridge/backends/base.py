"""Backend protocol.

Each provider in the TOML config instantiates one Backend implementation. The
dispatcher routes incoming requests to the right backend by parsing the
``model`` field as ``{provider_id}/{model_slug}``.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from dataclasses import dataclass, replace
from typing import Any

from ..errors import UnsupportedOperation

# Async callback invoked once when the upstream backend assigns a job/prompt id.
# The bridge persists this so video jobs can be cross-referenced for debugging.
UpstreamIdCallback = Callable[[str], Awaitable[None]]


@dataclass(frozen=True, slots=True)
class ModelEntry:
    """One row in the response of `GET /v1/models` for this backend.

    The dispatcher prefixes ``id`` with the provider's id before returning to
    the client. ``kind`` is "image", "video", "chat", "embedding", or None
    when the backend can't tell (e.g. an OpenAI-compat upstream that lists
    every model uniformly without modality hints).

    ``supports_tools`` is a non-standard extension surfaced for gateway-aware
    frontends — the bridge knows per-backend (and sometimes per-model) which
    models accept the OpenAI ``tools`` array. ``None`` means the backend
    didn't say; the client can fall back to its own per-endpoint default.

    ``context_window`` is likewise non-standard: the model's max context size
    in tokens, when the upstream exposes it (llama.cpp's ``meta.n_ctx`` /
    router ``--ctx-size``, vLLM's ``max_model_len``). ``None`` when unknown —
    the OpenAI ``/v1/models`` row carries no such field, so a frontend showing
    a "N / max tokens" budget falls back to its own config.

    ``capabilities`` is a non-standard extension listing the operations a model
    accepts, in ``{input}-to-{output}`` form: ``("text-to-image",)``,
    ``("text-to-image", "image-to-image")``, ``("image-to-video",)``, and so on.
    ``None`` when the backend didn't say. It exists because a model id no longer
    tells you: where a backend serves one model's text-driven and
    reference-image-driven halves as separate endpoints, the bridge may present
    them as a single model — at which point nothing in the name distinguishes
    "text only" from "also accepts a reference image", and a client would learn
    the difference from a failed request. A frontend can use this to enable or
    grey out image attachment per model.

    ``prompt_style`` and ``prompt_hint`` are non-standard extensions for image
    and video models, consumed by a gateway-aware frontend's prompt-enhancement
    pass. ``prompt_style`` is the prompt FORMAT this model prefers — for images
    e.g. "natural-language", "booru-tags", "keyword-soup", "hybrid"; for video
    e.g. "cinematic-prose", "structured-cinematic". ``prompt_hint`` is a freeform
    per-model nudge (a quality-tag prefix, a length cap, an audio-cue reminder,
    …). Both ``None`` when unset; for a ComfyUI workflow they come from the
    companion ``meta.json``. The bridge passes them through verbatim regardless
    of ``kind`` — it does not validate the vocabulary; that's the frontend's job.
    """

    id: str
    kind: str | None = None
    display_name: str | None = None
    supports_tools: bool | None = None
    context_window: int | None = None
    prompt_style: str | None = None
    prompt_hint: str | None = None
    capabilities: tuple[str, ...] | None = None


# Canonical ordering for capability strings, so the field is stable across
# backends and across runs regardless of how an upstream ordered its metadata.
_INPUT_MODALITY_ORDER: tuple[str, ...] = ("text", "image", "video", "audio")

# ModelEntry.kind is an output *category*; capability strings name the output
# modality itself, so a chat model reads "image-to-text" rather than
# "image-to-chat".
KIND_OUTPUT_MODALITY: dict[str, str] = {
    "image": "image",
    "video": "video",
    "chat": "text",
    "embedding": "embedding",
}


def make_capabilities(inputs: Iterable[str], output: str | None) -> tuple[str, ...] | None:
    """Build a ``ModelEntry.capabilities`` tuple from raw modality names.

    Only names in the known modality vocabulary are used; anything else is
    ignored, because upstream "inputs" metadata is not always purely modalities
    — ImageRouter's map mixes them with parameter flags
    (``{"text": true, "image": false, "mask": false, "quality": false}``), and
    passing those through would yield nonsense like ``"quality-to-image"``.

    ``None`` when the output modality is unknown or no input is recognised —
    the field's contract is that absence means "the backend didn't say", never
    "this model accepts nothing".
    """
    if not output:
        return None
    seen = set(inputs)
    ordered = [m for m in _INPUT_MODALITY_ORDER if m in seen]
    return tuple(f"{m}-to-{output}" for m in ordered) or None


# Lowercased inside a label, never at its start. Only "to" and "with" occur in
# fal's catalogue today; the rest are here so the rule reads as a rule.
_CONNECTORS: frozenset[str] = frozenset(
    {"to", "with", "and", "of", "for", "in", "on", "the", "a", "an", "from", "by"}
)

# Tokens whose conventional casing isn't title case. Purely cosmetic: an
# acronym missing from this map renders title-cased, which is unremarkable
# rather than wrong, so the table can lag the catalogue without consequence.
_ACRONYMS: dict[str, str] = {
    "ocr": "OCR",
    "sdxl": "SDXL",
    "lcm": "LCM",
    "svd": "SVD",
    "sam": "SAM",
    "rle": "RLE",
    "svg": "SVG",
    "mlsd": "MLSD",
    "hed": "HED",
    "pidi": "PiDi",
    "teed": "TEED",
    "bbox": "BBox",
    "lora": "LoRA",
    "controlnet": "ControlNet",
    "flf2v": "FLF2V",
    "srpo": "SRPO",
    "rf": "RF",
    "sd": "SD",
    "ai": "AI",
    "hd": "HD",
    "api": "API",
    "nsfw": "NSFW",
    "3d": "3D",
    "2d": "2D",
}

# A version-ish token — "v2.2", "a14b", "5b", "q3", "i2v", "o1", "2511".
# Uppercased whole, because these read as designators rather than words.
_VERSION_TOKEN = re.compile(r"^[a-z]{0,2}[\d.]+[a-z]{0,3}$")

# A scan-line resolution, where the trailing "p" is conventionally lowercase.
# Checked before _VERSION_TOKEN, which would otherwise yield "480P".
_RESOLUTION_TOKEN = re.compile(r"^\d+p$")


def _prettify_word(word: str, *, first: bool) -> str:
    if word in _ACRONYMS:
        return _ACRONYMS[word]
    if _RESOLUTION_TOKEN.match(word):
        return word
    if _VERSION_TOKEN.match(word):
        return word.upper()
    if word in _CONNECTORS and not first:
        return word
    return word[:1].upper() + word[1:]


def _prettify(suffix: str) -> str:
    """Render an id fragment as a label: ``v2.2-a14b/text-to-video`` to
    ``V2.2-A14B Text-to-Video``.

    Path separators become spaces; hyphens are kept, so compound terms stay
    visibly compound ("Text-to-Video", "Face-to-Full-Portrait") instead of
    dissolving into loose words.
    """
    segments: list[str] = []
    first = True
    for segment in suffix.split("/"):
        words: list[str] = []
        for word in segment.split("-"):
            if not word:
                continue
            words.append(_prettify_word(word, first=first))
            first = False
        segments.append("-".join(words))
    return " ".join(s for s in segments if s)


def _squashed(text: str) -> str:
    """Casefolded alphanumerics only, for comparing a label against a name."""
    return "".join(c for c in text.casefold() if c.isalnum())


def disambiguate_display_names(
    entries: list[ModelEntry], *, pinned: frozenset[str] = frozenset()
) -> list[ModelEntry]:
    """Make each entry's ``display_name`` unique by appending its id's variant path.

    Upstreams title an endpoint after the model *family*, not the endpoint: fal
    serves Florence-2 Large as eight task endpoints
    (``fal-ai/florence-2-large/object-detection``, ``/ocr-with-region``, …) and
    titles all eight ``Florence-2 Large``. The ids stay distinct and every one
    is separately callable, but a frontend that renders ``display_name`` — which
    is the point of the field — shows eight identical rows and the user cannot
    tell which is which. Nearly a third of fal's catalogue collides this way.

    Disambiguation appends only the part of the id that actually differs within
    the colliding group, so ``Wan`` becomes ``Wan (V2.7 Text-to-Image)`` rather
    than repeating the family segment every client already sees. The fragment is
    prettified rather than pasted in raw, since this field is what a picker
    renders; checked across fal's whole catalogue, prettifying merges no two
    distinct fragments, so the distinction survives the nicer spelling.

    Where the fragment merely restates the family (``Fooocus (Fooocus)``,
    ``Veo 3.1 (Veo3.1)``) the entry keeps its bare name: that endpoint is the
    family's base, and a parenthetical repeating the words to its left is noise.
    At most one member of a group can do this — fragments are unique within a
    group — so its siblings stay distinguished from it.

    ``pinned`` ids are never renamed — an operator who set ``display_name`` in
    config asked for that exact string, collision or not. They still take part
    in *detecting* a collision, so a catalogue entry that happens to clash with
    a pinned name is still pulled apart from it — the pin is honoured without
    letting it mask a duplicate.
    """
    groups: dict[str, list[int]] = {}
    for i, entry in enumerate(entries):
        if not entry.display_name:
            continue
        groups.setdefault(entry.display_name, []).append(i)

    out = list(entries)
    for name, idxs in groups.items():
        if len(idxs) < 2:
            continue
        renaming = [i for i in idxs if out[i].id not in pinned]
        if not renaming:
            continue
        # The shared prefix spans only the ids being renamed, not the whole
        # colliding group: a pinned id can be a strict prefix of its sibling
        # (``…/nano-banana-2`` vs ``…/nano-banana-2/lite``), and including it
        # would halt the scan early to preserve a segment for an entry that is
        # never rewritten — leaving the sibling labelled with the family name
        # it already collides on.
        segments = [out[i].id.split("/") for i in renaming]
        # Whether one member may keep the bare family name. Only when nothing
        # in the group is pinned: a pinned entry is *precisely* what already
        # holds that name, so letting a second entry fall back to it would
        # recreate the collision this function exists to remove.
        may_shed = len(renaming) == len(idxs)
        # With shedding allowed the scan may consume the shortest id entirely,
        # leaving an empty fragment — the base endpoint of a family whose
        # siblings extend its id (``…/framepack`` beside ``…/framepack/flf2v``).
        # The point is the siblings: they stop repeating the family segment,
        # giving "Framepack (FLF2V)" rather than "Framepack (Framepack FLF2V)".
        # Ids are unique, so the scan stops the moment the shortest runs out and
        # at most one fragment empties. Otherwise it halts a segment earlier, so
        # every fragment stays non-empty.
        floor = 0 if may_shed else 1
        shared = 0
        while (
            all(len(s) > shared + floor for s in segments)
            and len({s[shared] for s in segments}) == 1
        ):
            shared += 1
        for i, segs in zip(renaming, segments, strict=True):
            label = _prettify("/".join(segs[shared:]))
            # An empty fragment, or one that merely restates the family
            # ("Fooocus (Fooocus)"), leaves the bare name standing.
            if may_shed and (not label or _squashed(label) == _squashed(name)):
                continue
            out[i] = replace(out[i], display_name=f"{name} ({label})")
    return out


@dataclass(frozen=True, slots=True)
class GeneratedAsset:
    """The result of any generation: bytes + content type + kind."""

    data: bytes
    content_type: str
    kind: str  # "image" | "video"


@dataclass(frozen=True, slots=True)
class InputImage:
    """One reference image supplied to an image-edit request."""

    data: bytes
    content_type: str


class Backend(ABC):
    """Async backend protocol. Implementations are *not* expected to be thread-safe;
    they live for the duration of the bridge process and serve all requests for
    a single configured provider."""

    @abstractmethod
    async def list_models(self) -> list[ModelEntry]: ...

    async def generate_image(
        self,
        *,
        model_slug: str,
        prompt: str,
        size: str | None = None,
        n: int = 1,
    ) -> list[GeneratedAsset]:
        """Default: not supported. Override in backends that do text-to-image.

        Originally abstract — relaxed when the OpenAI-passthrough backend
        landed (chat/embedding upstreams have no image surface, but the
        Backend ABC is a single union of all backend capabilities).
        """
        raise UnsupportedOperation("Image generation is not supported by this provider")

    async def edit_image(
        self,
        *,
        model_slug: str,
        prompt: str,
        images: list[InputImage],
        size: str | None = None,
        n: int = 1,
    ) -> list[GeneratedAsset]:
        """Default: not supported. Override in backends that do img2img.

        ``images`` carries one or more reference images in client order.
        Backends that can forward multiples (ImageRouter's ``image[]``,
        OpenRouter's per-image content parts, ComfyUI multi-input workflows)
        pass the whole list through. The invariant is that a backend which
        can't use every supplied image must raise rather than silently drop
        one — either by letting the upstream reject it (ImageRouter) or by
        erroring at the bridge (ComfyUI, when a workflow has fewer image
        slots than images supplied).
        """
        raise UnsupportedOperation("Image edits are not supported by this provider")

    async def generate_video(
        self,
        *,
        model_slug: str,
        prompt: str,
        size: str | None = None,
        seconds: float | None = None,
        input_reference: bytes | None = None,
        input_reference_content_type: str | None = None,
        on_upstream_id: UpstreamIdCallback | None = None,
    ) -> GeneratedAsset:
        """Default: not supported. Override in backends that produce video.

        ``on_upstream_id`` is awaited once with the upstream's job id (e.g.
        ComfyUI's prompt_id) as soon as it's known, so the runner can persist
        it to the video_jobs row for resume/debug.
        """
        raise UnsupportedOperation("Video generation is not supported by this provider")

    # --- chat / embedding (OpenAI-passthrough territory) ----------------

    async def chat_completion(
        self,
        body: dict[str, Any],
        *,
        stream: bool,
    ) -> dict[str, Any] | AsyncIterator[bytes]:
        """Forward an OpenAI-shaped chat completion request to the backend.

        Default: not supported. Override in OpenAI-passthrough backends.

        When ``stream=False``, returns the upstream's parsed JSON response.
        When ``stream=True``, returns an async iterator of raw SSE byte chunks
        the bridge will pipe straight to the client without re-parsing — so a
        client's typewriter UI sees tokens land as the upstream emits them.
        The opaque-bytes shape on the streaming path is deliberate: chat
        completions chunks include vendor extensions (function calls, vision,
        tool outputs, JSON mode) we don't need to understand to forward.
        """
        raise UnsupportedOperation("Chat completions are not supported by this provider")

    async def create_embedding(
        self,
        body: dict[str, Any],
    ) -> dict[str, Any]:
        """Forward an OpenAI-shaped embeddings request. Default: not supported."""
        raise UnsupportedOperation("Embeddings are not supported by this provider")

    # --- lifecycle -------------------------------------------------------

    async def aclose(self) -> None:
        """Optional cleanup hook (e.g. close a shared httpx client)."""
        return
