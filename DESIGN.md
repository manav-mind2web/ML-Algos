# Window/Door Detection — Template-Matching Pipeline (Plan v2)

## Goal

Replace the lossy HF detector with a deterministic, near-free, sub-second
pipeline that uses the **type templates already in the payload** as the
detection signal. Claude vision runs only as a fallback verifier for
ambiguous cases.

Inputs are unchanged: the HF payload already contains
`building_image`, `window_types[]`, `door_types[]`, `garage_types[]`.

Output is the same `response.json` schema you have today, so nothing
downstream changes.

## End-to-end flow

```
payload.json
   │  building_image (1477×385)
   │  window_types[0..1]   (~80×60, RGBA)
   │  door_types[0]        (~55×50, RGBA)
   │  garage_types[0]      (~223×102, RGBA)
   │
   ▼
[1] Pre-process
       • decode all images, grayscale + Canny edges
       • build mask from each template's alpha channel
       • record per-template native size
   │
   ▼
[2] Multi-scale template matcher (cv2.matchTemplate, TM_CCOEFF_NORMED)
       for each template T:
         for scale s in 0.70 .. 1.40 step 0.05:
           score_map = matchTemplate(building_edges, resize(T_edges, s),
                                      mask=resize(T_mask, s))
           collect (x, y, s, score) where score ≥ τ_T
   │
   ▼
[3] Per-class NMS (IoU thresholds: windows 0.4, garage 0.4, doors 0.3)
       merge candidates from all scales+templates of the same class
       winning template name → `type`
   │
   ▼
[4] Coverage check
       compute density grid 256×256
       find "suspect regions": image areas with strong edge content
       but no detection inside
   │
   ▼
[5] Claude verifier  (only on residuals — usually 0–3 calls per image)
       (a) crop each suspect region → ask: "is there a window/door here?
            if yes, give a bbox in crop coords + type"
       (b) for low-score matches (0.55–0.75 band), crop and ask:
            "confirm this is class X, type Y"
       (c) on type ties (two templates scored within 0.03), crop and ask:
            "which of these two reference crops does this look like?"
   │
   ▼
[6] Assemble response.json (same schema as today)
       + review_meta { tier1_count, tier2_calls, elapsed_ms, ... }
```

Tier 1 is expected to handle 90–95% of detections at ~50 ms per image,
fully deterministic. Tier 2 fires only when Tier 1 is uncertain.

## Why this beats the alternatives for *this* data

| Approach              | Cost/img | Latency | Uses templates? | Notes                                   |
|-----------------------|----------|---------|-----------------|------------------------------------------|
| HF detector (current) | $        | ~1s     | no              | Misses things — the problem we're solving |
| Claude vision only    | $$$      | ~10s    | no              | Expensive, slow                          |
| SAM 2 + Claude        | $$       | ~5s     | no              | Returns masks for everything; needs filtering |
| **Template matching + Claude verifier** | ~$0 | ~0.1s | **yes** | This plan                                |

## Repo layout

```
ML-Algos/
├── DESIGN.md                       (this file)
├── pyproject.toml
├── src/
│   └── detect/
│       ├── __init__.py
│       ├── config.py               # thresholds, scale range, NMS IoU
│       ├── schema.py               # pydantic models, matches response.json
│       ├── payload.py              # decode payload.json → numpy arrays
│       ├── preprocess.py           # grayscale + edges + masks
│       ├── matcher.py              # multi-scale matchTemplate + NMS
│       ├── coverage.py             # suspect-region finder for Tier 2
│       ├── verifier.py             # Claude vision calls (Tier 2)
│       ├── prompts.py
│       └── assemble.py             # → response.json
├── tests/
│   ├── fixtures/
│   │   └── sample_payload.json     # the one you uploaded
│   ├── test_matcher.py
│   ├── test_nms.py
│   └── test_end_to_end.py
└── examples/
    └── run_detect.py               # CLI: payload.json → response.json
```

## Algorithm specifics

### Edge-based matching (not pixel-based)

Raw pixel match on architectural drawings is brittle to anti-aliasing
and stroke width. Pipeline:

1. `building_gray = cv2.cvtColor(building, COLOR_RGB2GRAY)`
2. `building_edges = cv2.Canny(building_gray, 50, 150)`
3. For each template:
   - extract alpha mask if RGBA, else binarize via Otsu
   - `T_edges = cv2.Canny(T_gray, 50, 150)`
4. Match `T_edges` against `building_edges` using `TM_CCOEFF_NORMED` with
   the resized mask passed as the `mask=` arg (so transparent template
   pixels don't contribute to the correlation).

This makes matching invariant to fill color and small stroke variations
while still being fully deterministic.

### Scale sweep

- Range: `0.70 → 1.40`, step `0.05` (15 scales).
- Justification: your sample shows building windows at ~99×75 vs
  template 81×62 → 1.21× scale. ±30% covers the realistic range for the
  same drawing program output. Configurable in `config.py`.
- For each scale, keep all peaks above `τ_T` (default `0.70`).

### NMS

- Per class, across all templates and scales.
- IoU thresholds: `windows=0.4`, `garage_doors=0.4`, `doors=0.3` (doors
  are narrow; 0.5 mis-merges adjacent doors).
- On overlap, keep the higher-scoring candidate. Its winning template
  becomes the detection's `type`.

### Coverage / suspect-region finder

After NMS, build a binary mask of "covered" pixels (the union of
detected bboxes, dilated by 10 px). Find connected components of
high-edge-density pixels (Canny output, dilated by 5 px) that lie
**outside** the covered mask. Components with area > 1500 px² → suspect
regions sent to Tier 2.

This is what catches openings the templates don't cover.

### Tier 2 — Claude verifier

Three call types, all with `cache_control` on the building image so the
image uploads once per request batch:

1. **Suspect-region check** — `crop = building[region]`, prompt: *"Is
   this crop a window, door, garage door, or none of those? If yes,
   bbox within the crop and short reason."*
2. **Low-score confirm** — for each Tier-1 candidate with score in
   `[0.55, 0.75]`, crop ±10 px and ask: *"Confirm class=X, type=Y."*
3. **Type tie-break** — when two templates score within 0.03, send the
   crop + both template crops and ask: *"Which template does this
   match — A or B?"*

Calls run in `asyncio.gather` bounded by `Semaphore(8)`. Hard cap of
10 Tier-2 calls per image, logged.

## Config defaults (`config.py`)

```python
SCALE_MIN, SCALE_MAX, SCALE_STEP = 0.70, 1.40, 0.05
MATCH_THRESHOLD = 0.70                    # τ_T
LOW_SCORE_BAND  = (0.55, 0.75)            # → Tier 2 confirm
TIE_DELTA       = 0.03                    # → Tier 2 type tie-break
IOU_NMS         = {"window": 0.4, "garage_door": 0.4, "door": 0.3}
SUSPECT_MIN_AREA = 1500
TIER2_MAX_CALLS  = 10
TIER2_MODEL      = "claude-haiku-4-5-20251001"   # cheap is enough
```

Thresholds are tuned on the test fixtures; never hardcoded inline.

## response.json schema (additive)

Same shape as today, plus per-detection provenance:

```json
{
  "id": 1,
  "bbox": [173, 151, 272, 226],
  "confidence": 0.91,
  "class": "window",
  "type": "Type 1",
  "source": "template_match" | "claude_verify" | "claude_add",
  "match_score": 0.87,
  "scale": 1.20
}
```

Plus a top-level `review_meta`:

```json
"review_meta": {
  "pipeline": "template_match_v1",
  "tier1_candidates": 14,
  "tier1_kept": 12,
  "tier2_calls": 1,
  "tier2_added": 0,
  "elapsed_ms": 142,
  "input_sha256": "..."
}
```

## Test plan

Using the sample payload you uploaded (4 windows, 4 doors, 4 garage doors
expected), the harness asserts:

- `test_matcher`: each template fires ≥1 hit on the sample, no class
  cross-contamination (window template doesn't match a garage door).
- `test_nms`: 60 raw candidates collapse to ≤16 detections.
- `test_end_to_end`: final `response.json` matches the HF response
  within IoU 0.5 per detection, **for all 12 expected objects**, and
  has zero false positives.

Tier 2 is mocked in tests; a separate `test_tier2_integration.py`
hits the real Claude API and is opt-in via env var.

## Milestones

1. **M1 — Payload decode + preprocess.** `payload.py`, `preprocess.py`,
   tests load the fixture and dump grayscale/edge images for visual
   inspection. *(½ day)*
2. **M2 — Multi-scale matcher + NMS.** `matcher.py`. Run on fixture,
   visualize all 12 expected detections. *(1 day)*
3. **M3 — Assemble response.json.** `assemble.py`, schema validation,
   end-to-end test green on fixture. *(½ day)*
4. **M4 — Coverage check + Tier 2 verifier.** `coverage.py`,
   `verifier.py`, prompt-tuning on 1–2 hard examples. *(1 day)*
5. **M5 — Eval harness.** Score on 20 hand-labelled images vs current
   HF baseline. Track precision/recall per class. *(1 day)*
6. **M6 — CLI + packaging.** `examples/run_detect.py`, README, ready to
   wire into your service. *(½ day)*

M2 alone should already match or beat the HF baseline on simple
elevations. Tier 2 is what closes the gap on edge cases.

## Failure modes & guards

| risk                                          | guard                                                                |
|-----------------------------------------------|----------------------------------------------------------------------|
| Window drawn outside 0.7–1.4× template scale  | Coverage check catches it as suspect region → Tier 2.                |
| Future image has a window type not in payload | Same — Tier 2 sweeps for uncovered openings.                          |
| Alpha channel is opaque (no real mask)        | Fallback to Otsu binarization of template; logged.                    |
| Template matches a shutter or roof tile       | Tier 2 low-score confirm rejects it before it enters response.json.   |
| Top-view image sneaks in                      | Single Claude call at start classifies view; refuse if `top`.         |
| matchTemplate OOM on huge images              | Tile building image into 1024-wide strips with 200 px overlap.        |

## What this plan deliberately omits

- No SAM 2, no Grounding DINO, no fine-tuning. The templates make those
  unnecessary at this scale.
- No replacement of the HF endpoint; the new pipeline runs *instead of*
  calling it. The old reviewer-on-top-of-HF plan in git history
  (`DESIGN.md` at commit `1ceeaa1`) is superseded.

## Open questions

1. Is the alpha channel of the template crops a true mask (transparent
   background) or is it fully opaque RGBA?
2. Are templates always supplied per request, or sometimes missing?
3. How many distinct types does each class actually have in production
   (the sample has only Type 1 for everything)?
