# Content Hub DAM – automatic object extraction

On every image upload to Sitecore Content Hub DAM, detect the distinct objects in it and create one new asset per object
(transparent-background PNG cutout), linked back to the source asset — ready to be fed into Gen-AI compositing.

## Architecture

```
Upload (UI / API)
   │
   ▼
M.Asset created / MainFile changed
   │  Trigger  (conditions: IsDerivedCutout != true)
   ▼
Action: Azure Service Bus  ──►  queue: ch-object-extraction
                                   │
                                   ▼
                     GPU worker (container, this repo)
                       1. GET /api/entities/{id}, wait for downloadOriginal
                       2. Grounding DINO  → open-vocabulary boxes
                       3. NMS + size filters
                       4. SAM             → pixel masks → RGBA cutouts
                       5. Upload API v2 (request → process → finalize) → new M.Asset per object
                       6. PUT props + ParentAssetToCutoutAsset relation
                       7. PUT status on the parent
```

Why external, not an M.Action script: the Content Hub scripting sandbox blocks outbound HTTP, and the work needs a GPU.
Why not Azure AI Vision: Image Analysis 4.0 background removal/segment was retired in 2025 and the whole service is
deprecated (retires 25 Sep 2028). Object detection alone gives rectangles; Gen-AI compositing needs real masks.

## 1. Content Hub schema (Manage → Schema → M.Asset)

Create a member group `ObjectExtraction` on **M.Asset**:

| Name | Type | Notes |
|---|---|---|
| `IsDerivedCutout` | Boolean | Loop guard + search facet |
| `CutoutLabel` | String | Detected class (bottle, box, …) |
| `CutoutConfidence` | Decimal | Detector score |
| `CutoutBoundingBox` | String (JSON) | `{x0,y0,x1,y1,kind}` in source pixels |
| `ObjectExtractionStatus` | String (or Option list: Processing / Completed / NoObjects / Failed) | On the parent |
| `ObjectExtractionCount` | Integer | On the parent |
| `ObjectExtractionMessage` | String | Error / created IDs |
| `ObjectExtractionSourceHash` | String | Idempotency – SHA-256 of processed original |

Relation **`ParentAssetToCutoutAsset`**: M.Asset → M.Asset, **OneToMany**, parent role on the source asset.
(If you pick a different cardinality, change the relation payload in `main.py`.)

Mark `ObjectExtraction*` properties as non-editable in the asset detail page; show the relation as a
"Cutouts" list component on the parent's detail page and a "Source asset" link on the child.

## 2. Azure

1. Service Bus namespace + queue `ch-object-extraction`: lock duration 5 min, MaxDeliveryCount 5, dead-lettering on.
2. Two SAS policies: **Send** (for Content Hub) and **Listen** (for the worker).
3. GPU host: Azure Container Apps (serverless GPU, T4/A100 workload profile) scaled by KEDA on queue length, min 0.
   Rough sizing: grounding-dino-base + sam-vit-large ≈ 6–8 GB VRAM; 2–6 s per image on a T4.

## 3. Content Hub action + trigger

**Manage → Actions → New action**
- Type: **Azure Service Bus**
- Connection string: the *Send* SAS; Destination type: Queue; Name: `ch-object-extraction`

**Manage → Triggers → New trigger**
- Objective: **Entity creation** *and* **Entity modification**
- Execution type: **In background**
- Conditions (definition M.Asset):
  - `MainFile` **has changed**   *(fires on new upload and new version)*
  - `IsDerivedCutout` **is not** `true`
- Actions: the Service Bus action above. Activate.

**Manage → OAuth clients**: create a client for the worker; use a dedicated integration user in a group that can
Create/Read/Update M.Asset and upload to the target repository.

## 4. Run the worker

```bash
cd worker
docker build -t ch-object-extraction .
docker run --gpus all \
  -e CH_BASE_URL=https://<tenant>.sitecorecontenthub.cloud \
  -e CH_CLIENT_ID=... -e CH_CLIENT_SECRET=... \
  -e CH_USERNAME=svc-extract -e CH_PASSWORD=... \
  -e SB_CONNECTION="Endpoint=sb://...;SharedAccessKeyName=listen;SharedAccessKey=..." \
  ch-object-extraction
```

Tune on your own photography first, without touching Content Hub:

```bash
python local_test.py samples/flatlay.jpg out/   # writes cutouts + _preview.jpg with boxes
```

### Key settings (env vars, see `config.py`)

| Var | Default | Effect |
|---|---|---|
| `DETECT_LABELS` | product,bottle,box,package,tube,jar,can,pouch,blister pack,device,logo | What Grounding DINO looks for. Make this your client's packaging vocabulary. |
| `BOX_THRESHOLD` / `TEXT_THRESHOLD` | 0.35 / 0.25 | Raise to cut false positives |
| `NMS_IOU` | 0.5 | Duplicate collapse across labels |
| `MIN_AREA_RATIO` / `MAX_AREA_RATIO` | 0.01 / 0.90 | Drops specks and "the whole frame" |
| `MIN_OBJECTS_TO_SPLIT` | 2 | Single-product packshots are left alone |
| `MAX_OBJECTS` | 10 | Cap per image |
| `OUTPUT_MODE` | cutout | `cutout` (RGBA PNG), `crop` (rectangle), `both` |

## Behaviour notes

- **Loop prevention** – three layers: trigger condition, `IsDerivedCutout` check, `__cutout_` filename marker
  (the trigger can fire before properties are written on a new child).
- **Idempotency** – Service Bus is at-least-once and modification triggers fire repeatedly; the worker skips when
  the original's SHA-256 matches `ObjectExtractionSourceHash`.
- **New version of the parent** – produces a fresh set of cutouts; old ones are *not* deleted (do that via
  a reviewed bulk action if needed).
- **Lifecycle** – children are created in the default state (Created). If they should inherit the parent's
  approval, also set `FinalLifeCycleStatusToAsset` in the child PUT, and copy taxonomy relations
  (brand, product, campaign) from the parent there too.
- **Failures** surface on the parent (`ObjectExtractionStatus = Failed` + message) and the message is
  dead-lettered after 5 attempts.

## Gen-AI handoff

Cutouts are query-able as a set: `IsDerivedCutout = true AND CutoutLabel = 'bottle'`, or via the parent relation.
Transparent PNGs drop straight into inpainting/compositing pipelines (SitecoreAI, Firefly, SDXL/Flux inpaint)
as the product layer; keep `CutoutBoundingBox` to re-place the product at its original scale.

**Rights:** compositing products with real, identifiable celebrities needs signed likeness/endorsement rights for
each person, brand and usage, and most commercial image-model terms forbid generating real people without it.
Model this in the DAM, not in people's heads: a `Talent` entity with contract scope/expiry, related to every
generated asset, and a lifecycle gate that blocks approval when the relation is missing or expired.
