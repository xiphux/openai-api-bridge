# The `aspect_ratios` field

Client-facing spec for the non-standard `aspect_ratios` /
`aspect_ratio_default` entries on `GET /v1/models`, and the `aspect_ratio`
request parameter they enable. Written for frontends that want to offer a
shape picker beside the prompt box.

For the operator-facing view of this and the other non-standard fields, see
[Model metadata extensions](../README.md#model-metadata-extensions-nonstandard)
in the README; for the workflow side, the
[meta.json schema](../README.md#workflow-metajson-schema-comfyui).

## Why it exists

OpenAI's images API sizes an output in pixels: `size: "1792x1024"`. That
transfers badly to a self-hosted box.

A user thinking "widescreen" has to do arithmetic to say so, and the arithmetic
is model-specific — the megapixel count a 12GB card can hold differs from a
24GB one, and both differ per architecture. Worse, pixels are an *unbounded*
knob pointed at a single GPU: nothing stops a client asking for 4096×4096 and
taking the box down with it.

An aspect ratio inverts that. The operator fixes the pixel budget once, in the
workflow, sized to the hardware. The client picks a shape, and every shape
renders inside the same budget. So the knob a user actually wants is the only
knob exposed, and the one that could hurt is absent rather than validated.

Several upstreams already work this way natively — fal's Nano Banana / Gemini
image models take `aspect_ratio` and reject `image_size` outright — so this is
not only a ComfyUI convenience.

## Shape

Two optional fields on a model row:

```json
{
  "id": "comfyui/minimax-h3-t2v",
  "kind": "video",
  "aspect_ratios": [
    { "value": "1:1", "label": "Square" },
    { "value": "2:3", "label": "Portrait Photo" },
    { "value": "16:9", "label": "Widescreen" },
    { "value": "21:9", "label": "Ultrawide" }
  ],
  "aspect_ratio_default": "16:9"
}
```

### Guarantees

* **Order is meaningful.** It's the order the upstream declared, intended as
  render order. Don't sort it.
* `value` is `W:H`, both positive integers, no spaces. **It is not reduced** —
  `21:9` stays `21:9`, because that is the name people use (and true ultrawide
  is 64:27 regardless; `21:9` is a convention that stuck). Treat it as an
  opaque token you echo back, not a fraction to normalise.
* `label` is **optional** and freeform. Absent when the upstream's own option
  had no human name. Never render a placeholder for it — derive the icon from
  the two numbers instead, which also means an unfamiliar ratio needs no
  client change.
* Values are unique within a list.
* `aspect_ratio_default` is a `value` the model produces when a request names
  no ratio. It is **not guaranteed** to appear in `aspect_ratios` (a workflow
  can be saved at a ratio its menu doesn't list), so don't assume you can
  pre-select it.
* Both fields are **omitted, never null**, when the upstream didn't say.

## Consuming it

### Absence means "no selector", not "one fixed ratio"

Same rule as `capabilities`. A model without `aspect_ratios` may still produce
whatever shape it likes — the bridge just has nothing to tell you about it.
Hide the picker; don't grey it out, and don't assume 1:1.

### Send the `value` verbatim

`POST /v1/images/generations` takes `aspect_ratio` in its JSON body:

```json
{ "model": "comfyui/ratio-t2i", "prompt": "a lighthouse", "aspect_ratio": "16:9" }
```

`POST /v1/images/edits` and `POST /v1/videos` are multipart, so there it is a
form field named `aspect_ratio` alongside the others. Sending it as JSON to
either of those is silently ignored, the same as any other field would be.

Only send it to a model that advertised ratios. A model that didn't will
ignore it rather than erroring, so nothing breaks — but a generic
OpenAI-compatible upstream behind the bridge is a different matter, and the
advertisement is how you know you're not talking to one.

### `aspect_ratio` and `size` are alternatives

A model takes at most one. If it advertises `aspect_ratios`, its size is fixed
upstream and `size` is ignored for it. Don't send both; if you do, the ratio
wins for such a model and `size` wins for every other.

### A ratio you send need not be the one you get

**This is the part worth reading twice.** If a model doesn't offer the exact
ratio you asked for, the bridge picks the **nearest ratio it does offer**,
measured in log space so `2:1` is as far from `1:1` as `1:2` is. It does not
return a 400, and it does not fall back to the model's own default.

(A value that isn't a `W:H` token at all — `"wide"`, `null`, a number — is not
"a ratio you sent" and gets no snapping: it is treated as naming nothing, so the
model's own default stands.)

That matters because a client's menu and a model's menu drift apart for
ordinary reasons:

* a remembered "last used" preference, carried to a model with a different
  list;
* one prompt fanned out across several models at once;
* an operator swapping a workflow's resolution node.

Falling back to the model's default would be the worst of the three options —
ask for `21:9` and receive the workflow's baked-in `2:3` portrait, which is
further from the request than anything in the list. Snapping gives you `16:9`.

So **read the echo rather than recording your request**:

* Images: `aspect_ratio` on each `data[]` entry, **omitted** when the backend
  didn't deal in ratios.
* Video: `aspect_ratio` on the job object from `GET /v1/videos/{id}`. Seeded
  from the request at creation, then settled once the render completes — to
  what was rendered, or to `null` when the backend deals in no ratios. So read
  it off the **completed** job; on a queued one it is still just your request.

Note the two surfaces differ in shape, deliberately: the image field is absent
when there's nothing to say, while the job field is always present and may be
`null`. The job object carries `size` the same way, so `aspect_ratio` matches its
neighbours rather than the image echo. Handle both.

If you persist a generation's shape — to label it, or to reproduce it on a
regenerate — persist the echoed value.

### Suggested predicate

```ts
type AspectRatio = { value: string; label?: string };

/** Ratios to offer for a set of selected models: the union, in first-seen
 *  order. A model that offers none constrains nothing — it ignores whatever
 *  is sent, which is the same outcome as it not being selected. */
function offeredRatios(models: { aspect_ratios?: AspectRatio[] }[]): AspectRatio[] {
  const seen = new Map<string, AspectRatio>();
  for (const m of models)
    for (const r of m.aspect_ratios ?? []) if (!seen.has(r.value)) seen.set(r.value, r);
  return [...seen.values()];
}
```

Union rather than intersection, deliberately. Intersection shrinks the menu as
a user adds models — and would let a model with a *different* list restrict
everyone while a model with *no* list restricts nobody, which is backwards.
Snapping is what makes the union safe: each model resolves the chosen ratio
against its own menu.

## Which models carry it

Today: ComfyUI workflows whose `meta.json` declares `aspect_ratio_node` plus
`aspect_ratios`. Nothing else populates the field yet, though fal and Venice
both have models that would fit it.

## Stability

Additive and optional, like `capabilities`. New ratios may appear in a list,
`label` text may change, and a model may gain or lose the fields when an
operator edits a workflow. A client must not hard-code a vocabulary of ratios
or assume a fixed list length; render what the row says and send back one of
its `value`s.
