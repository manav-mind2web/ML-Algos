# Window/Door Detection Review Pipeline — Design

## Goal

Fix imperfect detections from the HF-hosted detector by having Claude Opus
vision review the image and **directly edit `response.json`** so that
re-annotating the raw image with the edited JSON produces a perfect overlay.

Scope: front, side, and back views of building exteriors. No top views.
Classes: `window`, `door`, `garage_door` (with `type` subfield).

## End-to-end flow

```
raw_image.png
   │
   ▼
[1] HF detector  ──────────────▶  response.json  (v0, imperfect)
   │                                   │
   ▼                                   ▼
[2] annotate(raw, v0)  ──────▶  annotated_v0.png
   │
   ▼
[3] Claude review agent (Opus vision)
       inputs : raw_image.png, annotated_v0.png, response.json (v0)
       tools  : crop_region, annotate_image, write_response_json
       output : response.json (v1, corrected)
   │
   ▼
[4] annotate(raw, v1)  ──────▶  annotated_v1.png   ← final deliverable
   │
   ▼
[5] self-check pass (Claude looks at annotated_v1.png only,
                     confirms every box is tight & no openings missed.
                     If not, loop back to [3] with max 2 retries.)
```

The agent **edits JSON, not pixels.** The annotated image is regenerated
deterministically from the JSON after every edit. This guarantees the
image and the JSON can never drift apart.

## Repo layout

```
ML-Algos/
├── DESIGN.md                       ← this file
├── pyproject.toml
├── src/
│   └── review/
│       ├── __init__.py
│       ├── config.py               ← thresholds, model IDs, paths
│       ├── schema.py               ← pydantic models for response.json
│       ├── hf_client.py            ← calls the HF endpoint
│       ├── annotate.py             ← raw + JSON → annotated PNG
│       ├── crop.py                 ← bbox → cropped PIL.Image
│       ├── prompts.py              ← all prompt strings
│       ├── agent.py                ← Claude Agent SDK orchestration
│       ├── tools.py                ← SDK tool definitions
│       └── reconcile.py            ← IoU diff + JSON merge helpers
├── tests/
│   ├── fixtures/                   ← sample image + response pairs
│   └── test_reconcile.py
└── examples/
    └── run_review.py               ← CLI entrypoint
```

## response.json schema (locked)

Same shape as the HF response you pasted. Three top-level groups
(`windows`, `doors`, `garage_doors`), each with `detections[]` and
`summary{}`. Every detection:

```json
{
  "id": 1,
  "bbox": [x1, y1, x2, y2],     // pixel coords, absolute
  "confidence": 0.87,
  "class": "window",            // window | door | garage_door
  "type": "Type 1",             // class-specific enum
  "source": "hf" | "claude_edit" | "claude_add",   // NEW, optional
  "edit_reason": "tightened bbox; was 8px too wide on the right"  // NEW, optional
}
```

`source` and `edit_reason` are **additive** — downstream consumers that
expect the old schema keep working. `summary` is recomputed by the
writer, never edited by hand.

## The Claude review agent

### Loop

```
state = load(response_v0)
for attempt in range(MAX_ATTEMPTS = 3):
    annotated = annotate(raw, state)
    verdict   = claude_review(raw, annotated, state)
    if verdict.status == "PERFECT": break
    state = apply_edits(state, verdict.edits)
save(state)
```

`verdict.status ∈ {PERFECT, NEEDS_EDIT}`. The agent stops as soon as
Claude says the annotated image is perfect, or after 3 attempts.

### Edit operations (the only things Claude can emit)

Forced via tool schema. Claude never returns free-form JSON.

| op             | fields                                              | meaning                                      |
|----------------|-----------------------------------------------------|----------------------------------------------|
| `adjust_bbox`  | `class`, `id`, `new_bbox`, `reason`                 | Tighten or shift an existing box.            |
| `change_class` | `class`, `id`, `new_class`, `new_type`, `reason`    | Reclassify (e.g. door→window).               |
| `change_type`  | `class`, `id`, `new_type`, `reason`                 | Keep class, fix the type label.              |
| `delete`       | `class`, `id`, `reason`                             | False positive.                              |
| `add`          | `class`, `bbox`, `type`, `reason`                   | Missed detection. `id` auto-assigned.        |

The reconciler applies edits in this order: `delete` → `adjust_bbox` →
`change_class` → `change_type` → `add`. New `id`s start at
`max(existing_ids) + 1` per group.

### Two-pass prompting

Single-shot "review this" tends to anchor on the existing boxes. So:

**Pass A — blind enumeration.** Claude sees *only* `raw_image.png`. It
outputs a list of every opening it sees with rough bboxes in pixel
coords and a one-line description. This is cached in agent state.

**Pass B — critique & edit.** Claude sees `raw_image.png`,
`annotated_v0.png`, `response.json (v0)`, and **its own pass-A list**.
It emits `edits[]` as above. Having pass A on hand makes "missed
detections" almost free to spot — it just diffs its own list against
the JSON.

**Pass C — self-check.** After edits are applied and the image is
re-annotated, Claude sees only `annotated_v1.png` and answers
`PERFECT` / `NEEDS_EDIT` with reasons. If `NEEDS_EDIT`, loop.

### Tiling for small windows

Architectural renders pack small dormer/transom windows that get lost
at full resolution. After Pass A, compute detection density per
512×512 tile. For any tile with ≥3 detections **or** any tile where
HF has 0 detections but Pass A has ≥1, re-run Pass A on the
cropped tile at native resolution. Map bboxes back via the tile
offset. Merge into the Pass A list with NMS at IoU 0.5.

This single trick is the biggest recall win. Budget: ~2–4 extra
Claude calls per image.

### Per-detection type classification

`type` ("Type 1", "Type 2", ...) is hard to get right alongside
detection. After all bbox edits are final, for every detection emit
a separate Claude call with just the cropped bbox + the enum of
allowed types for that class. Cheaper, more accurate, easy to cache.

## Reconciliation rules

Used by `apply_edits` and by tests.

- **Match threshold (IoU):** windows 0.5, garage_doors 0.5, doors 0.3
  (doors are narrow and adjacent doors get mis-matched at 0.5).
- **Confidence after Claude edit:** keep HF confidence unless Claude
  adjusted the bbox, in which case set `confidence = max(hf, 0.9)`
  (Claude-confirmed boxes are high-trust).
- **Claude-added detections:** `confidence = 0.85`, `source = "claude_add"`.
- **Deleted detections:** dropped entirely, not soft-flagged.
- **Summary recompute:** always derived from `detections` after edits.

## Prompts (key snippets)

### Pass A — blind enumeration

```
You are inspecting a {view} elevation drawing of a residential building.
List every visible window, door, and garage door. For each, output:
  - class: window | door | garage_door
  - bbox: [x1, y1, x2, y2] in pixel coordinates of the original image
  - one-line description (location + distinguishing feature)
Do not skip small openings (dormers, transoms, sidelights). Do not list
roof vents, skylights, or shutters as windows. Return strict JSON
matching the schema in the tool definition.
```

### Pass B — critique & edit

```
You previously enumerated openings (PASS_A_LIST below).
You now also see (1) the same image, (2) an annotated overlay produced
from the model's response.json, and (3) the response.json itself.

For each detection in response.json, decide:
  - keep as-is
  - adjust_bbox  (box is loose, shifted, or clipped)
  - change_class (e.g. labeled door but it's a window)
  - change_type
  - delete       (nothing there, or it's a shutter/vent/wall feature)

Then, for every item in PASS_A_LIST that has no matching detection in
response.json (IoU < {iou}), emit an `add` edit.

Output only an `edits[]` array via the submit_edits tool. No prose.
```

### Pass C — self-check

```
This annotated image was produced by applying your edits. Look at it
fresh. Is every box tight around an actual window/door/garage door,
with nothing missing? Reply via the verdict tool with PERFECT or
NEEDS_EDIT plus a short reason list.
```

## Cost & latency control

- **Prompt caching** on `raw_image.png` — cached once, reused across
  passes A/B/C and every per-bbox type call. With Opus this is the
  single biggest cost lever.
- **Pass A tile crops** are *not* cached (one-shot per tile).
- **Per-bbox type calls** run concurrently (asyncio.gather, bounded
  by a semaphore of 8).
- Hard cap: 3 review loops + tiling + N type calls = upper bound on
  spend per image, logged to `review_meta`.

## Failure modes & guards

| risk                                              | guard                                                       |
|---------------------------------------------------|-------------------------------------------------------------|
| Claude invents detections that aren't there       | Pass C self-check on the re-annotated image must confirm.   |
| Edit loop oscillates                              | `MAX_ATTEMPTS = 3`, then bail and keep last state.          |
| Bbox coords drift out of image bounds             | Clamp in `apply_edits`; reject zero-area boxes.             |
| Class enum violations                             | Pydantic validation on tool input rejects the edit.         |
| HF endpoint flaky                                 | Cache v0 response on disk keyed by image hash.              |
| Top-view image sneaks in                          | A 1-call view classifier up front; refuse if `top`.         |

## Observability

Every run writes `review_meta` into the final JSON:

```json
"review_meta": {
  "model": "claude-opus-4-7",
  "attempts": 2,
  "edits_applied": {"adjust_bbox": 3, "add": 2, "delete": 1},
  "tiles_inspected": 4,
  "final_verdict": "PERFECT",
  "elapsed_ms": 18234,
  "input_image_sha256": "..."
}
```

Plus, every loop iteration is dumped to `runs/{sha}/iter_{n}/` with
`annotated.png`, `response.json`, and the raw Claude tool calls. This
becomes the fine-tuning dataset for the HF model later.

## Milestones

1. **M1 — Skeleton + annotate.** Stubs land, `annotate(raw, json)`
   works end-to-end on the fixture you provided. *(half day)*
2. **M2 — Single-pass review.** Pass B only, no tiling, no self-check.
   Proves the SDK wiring and edit schema. *(1 day)*
3. **M3 — Three-pass loop.** Add Pass A + Pass C + retry loop. *(1 day)*
4. **M4 — Tiling.** Density-based tile selection + NMS merge. *(half day)*
5. **M5 — Per-bbox type classifier.** Concurrent calls + caching. *(half day)*
6. **M6 — Eval harness.** Score against a hand-labelled set of ~20
   images. Track precision/recall per class vs HF baseline. *(1 day)*

M2 alone should already beat the HF baseline noticeably; everything
after that is squeezing out the last 15–20% of recall.
