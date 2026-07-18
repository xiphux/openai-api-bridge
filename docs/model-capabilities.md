# The `capabilities` field

Client-facing spec for the non-standard `capabilities` entry on `GET /v1/models`.
Written for frontends that need to know, per model, whether a user may (or must)
attach an image.

For the operator-facing view of this and the other non-standard fields, see
[Model metadata extensions](../README.md#model-metadata-extensions-nonstandard) in
the README.

## Why it exists

A model id used to tell you what it accepted: fal published
`fal-ai/nano-banana-2` and `fal-ai/nano-banana-2/edit` as separate endpoints, and
the `/edit` suffix was the signal. The bridge now **collapses** such pairs into a
single id and routes a request carrying a reference image to the sibling
automatically, because two near-identical ids in a picker is an easy thing to get
wrong.

That removes the signal. Nothing in `fal/fal-ai/nano-banana-2` distinguishes
"text only" from "also accepts a reference image", and without `capabilities` a
client would learn the difference from a failed request.

## Shape

A JSON array of strings on each model row:

```json
{
  "id": "fal/fal-ai/nano-banana-2",
  "object": "model",
  "created": 1784414836,
  "owned_by": "fal",
  "display_name": "Nano Banana 2",
  "kind": "image",
  "capabilities": ["text-to-image", "image-to-image"]
}
```

Each string is `{input-modality}-to-{output-modality}`.

| Position | Values |
|---|---|
| Input | `text`, `image`, `video`, `audio` |
| Output | `image`, `video`, `text`, `embedding` |

### Guarantees

* **Inputs appear in a fixed order** — `text`, `image`, `video`, `audio` — stable
  across backends and across runs, regardless of how an upstream ordered its own
  metadata. Arrays can be compared directly without sorting.
* **The field is omitted when unknown.** It is never `null` and never `[]`. A
  backend that can't determine a model's inputs leaves the key out entirely.
* Every string uses the vocabulary above. Upstream metadata that mixes modalities
  with unrelated flags (ImageRouter's `inputs` map carries `mask` and `quality`
  alongside `text` and `image`) is filtered, so values like `quality-to-image`
  cannot appear.

### `capabilities` vs `kind`

`kind` is an output *category*; `capabilities` names the output *modality*. They
differ for chat models:

| `kind` | output modality in `capabilities` |
|---|---|
| `image` | `image` |
| `video` | `video` |
| `chat` | **`text`** |
| `embedding` | `embedding` |

So a vision chat model is `kind: "chat"` with `["text-to-text", "image-to-text"]`
— there is no `text-to-chat`. Don't derive either field from the other; a model
can carry `kind` with no `capabilities`, or the reverse.

## Values in practice

The full set observed across a live 752-model listing (predominantly fal):

| Count | `kind` | `capabilities` |
|---|---|---|
| 299 | `image` | `["image-to-image"]` |
| 128 | `image` | `["text-to-image"]` |
| 113 | `video` | `["image-to-video"]` |
| 83 | `image` | `["text-to-image", "image-to-image"]` |
| 75 | `video` | `["text-to-video", "image-to-video"]` |
| 54 | `video` | `["text-to-video"]` |

OpenRouter also emits `["text-to-text"]` and `["text-to-text", "image-to-text"]`
for chat models. `embedding` is in the vocabulary and reachable by construction,
but no live catalogue or test has produced it — treat it as possible, not
expected.

## Which models carry it

| Backend | Emits it | Source |
|---|---|---|
| fal | yes | catalogue category; only the four image/video combinations |
| ComfyUI | yes | the workflow's `image_inputs` declaration in `meta.json` |
| Venice | yes | its `type=image` / `type=inpaint` listings; image outputs only |
| ImageRouter | yes | the per-model `inputs` map |
| OpenRouter | yes | `architecture.input_modalities`; the only backend producing `text`/`embedding` outputs |
| OpenAI passthrough | **never** | generic OpenAI-compatible upstreams don't report modality |

Any backend also omits it for an individual model whose upstream metadata is
silent — an ImageRouter entry with no `inputs` map, say. **Presence is per-model,
not per-provider.** Do not assume that because one model from a provider has the
field, all of them do.

## Consuming it

### Absence is not denial

An omitted field means *"the upstream didn't say"*, never *"this model accepts
nothing"*. Fall back to your own per-endpoint default. A model without
`capabilities` must not render as unusable — for an OpenAI-passthrough provider
that would disable every model it serves.

### Image attachment is a tri-state

The useful question is not "may I attach an image" but "what does this model do
with one":

| Capabilities | Attachment | UI |
|---|---|---|
| `text-to-X` **and** `image-to-X` | optional | offer it, don't require it |
| only `image-to-X` | **required** | require an image before allowing submit |
| only `text-to-X` | not accepted | hide or grey out |
| *field absent* | unknown | your fallback |

**The second row is the one to build for.** In the 752-model listing above, **412
models — 55% — require an image**: 299 `image-to-image` only (upscalers,
background removal, inpainting, the Florence-2 vision tasks) and 113
`image-to-video` only. A client that treats the field as a boolean ("does it
contain `image-to-`?") makes attachment look optional on all 412, and every
text-only request against one fails upstream after the user has already committed
to a prompt.

### Suggested predicate

```ts
type Attachment = "required" | "optional" | "unsupported" | "unknown";

function imageAttachment(model: { capabilities?: string[] }): Attachment {
  const caps = model.capabilities;
  if (!caps?.length) return "unknown";
  const fromImage = caps.some((c) => c.startsWith("image-to-"));
  const fromText = caps.some((c) => c.startsWith("text-to-"));
  if (fromImage && fromText) return "optional";
  if (fromImage) return "required";
  return "unsupported";
}
```

Match on the `{input}-to-` prefix rather than the whole string, so a new output
modality doesn't silently fall through to `unsupported`.

## Stability

Additive and optional. The bridge may start emitting the field for models that
lack it today (as upstream metadata improves), and new `{input}-to-{output}`
combinations may appear from the vocabulary above. Treat an unrecognised
combination as informational rather than an error.
