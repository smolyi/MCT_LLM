# Multi-Camera Tracking + Event Graph + LLM Reasoning — Project Guide

## Goal

Build a system that answers compositional natural-language queries about
people/objects observed across multiple cameras -- e.g. "find a man in a
grey shirt who parked a red car and later crossed the street carrying a
blue backpack." This is a learning project (companion to `../LMM_dive`),
prioritizing understanding the real architecture of such systems over
building a production-grade one.

Core thesis (established through discussion, not assumed): flat
RAG/embedding retrieval alone cannot answer this class of query, because
it has no mechanism for (a) compositionality across many constraints,
(b) temporal/sequential event ordering, or (c) persistent entity identity
across time and cameras. The fix is a structured **event graph** as the
backbone, with embeddings/RAG used as a component for fuzzy attribute
matching at the leaves, not as the top-level search mechanism.

## Data ethics -- why this dataset, specifically

We evaluated several multi-camera tracking datasets and rejected the
most "standard" one (WildTrack) on ethical grounds: it recorded real
students via surveillance cameras without meaningful informed consent
(notices were only visible to people who approached the camera hardware
directly), and independent analysis (exposing.ai) draws a direct parallel
to DukeMTMC, a dataset that was retracted after the responsible
researcher's public apology. Market-1501/MSMT17 (standard ReID
benchmarks) carry similar concerns and are avoided for the same reason.

We use **`nvidia/PhysicalAI-SmartSpaces`** (the AI City Challenge MTMC
Tracking dataset) instead: primarily **synthetic** data (generated via
NVIDIA Omniverse/Cosmos Transfer), so there is no real-person consent
question at all. CC-BY-4.0 licensed, hosted on HF Hub. (Its 2026 edition
adds 2 real-world test scenes -- `Warehouse_026`/`Warehouse_027` -- which
we deliberately avoid; stick to the synthetic scenes.)

## Environment

- Separate from `../LMM_dive/lmm-env` -- this project has a different
  dependency set (object detection, tracking, graph libraries) and
  should not bloat that environment.
- GPU required for detection/tracking inference.

## Data source details

Dataset: `nvidia/PhysicalAI-SmartSpaces` on HF Hub, `MTMC_Tracking_2024`
edition, `test` split (has public ground truth; `train` split's labels
are organized differently and weren't needed for this project's scope).

Structure: `MTMC_Tracking_2024/test/scene_NNN/camera_CCCC/{video.mp4,
calibration.json}`, plus one `scene_NNN/ground_truth.txt` per scene
covering all its cameras. 30 test scenes have public ground truth
(scene_061 through scene_090), each with ~10 cameras, each camera's
video ~155MB.

**`ground_truth.txt` columns** (space-separated, no header):
```
camera_id  object_id  frame_id  bbox_left  bbox_top  bbox_width  bbox_height  world_x  world_y
```
`object_id` is globally consistent across cameras within a scene -- this
*is* the ground-truth cross-camera identity association, given for free.
`world_x`/`world_y` are real-world ground-plane coordinates (meters,
scene-relative origin) -- use these for spatial-plausibility reasoning
rather than re-deriving position from calibration ourselves where
possible.

**`calibration.json`**: `camera projection matrix` (3x4, standard
P = K[R|t]) and `homography matrix` (3x3, image-plane <-> ground-plane
mapping). Reprojection error is near-zero (synthetic calibration is
effectively exact).

First working scope: **scene_061**, a small subset of its ~10 cameras
(start with 3-4, not all 10) -- enough to demonstrate genuine
cross-camera behavior without downloading/processing the full scene.
Downloaded to `data/scene_061/`: `ground_truth.txt` (scene-wide) +
`camera_{0535,0536,0537,0538}/{video.mp4,calibration.json}`. Videos are
1920x1080 @ 30fps, ~24000 frames (~800s) each.

## Official evaluation code (reused, not reimplemented)

The dataset repo ships its own eval code at
`MTMC_Tracking_2024/eval/` (real TrackEval, MIT-licensed, from
github.com/JonathonLuiten/TrackEval) -- downloaded locally to
`eval_ref/` for reference. We use this rather than a hand-rolled metric,
consistent with this project's "use the real methodology" convention
(see `../LMM_dive`'s VQA/POPE eval for precedent).

Key confirmed facts about it:
- **Prediction file format is identical to `ground_truth.txt`**: 9
  space-separated columns, `camera_id object_id frame_id bbox_left
  bbox_top bbox_width bbox_height world_x world_y`. Critically, this
  means **`object_id` in our predictions must already be globally
  consistent across cameras** -- the eval script merges all of a
  scene's cameras into one sequence and matches purely on `Id` +
  world-coordinate distance. There is no separate "cross-camera
  matching" step in the eval itself; our pipeline has to produce that
  consistency before writing the prediction file.
- **Metric: HOTA computed via 3D (world-coordinate) Euclidean distance**,
  not image-space IoU (`trackeval/datasets/mot_challenge_3d_location.py`,
  `zero_distance=2.0`, match `THRESHOLD=0.5` -- i.e. matching happens in
  meters on the ground plane, not pixels). This means our pipeline must
  convert each detection's image bbox into world coordinates using
  `calibration.json`'s homography matrix (project e.g. the bbox
  bottom-center -- the ground-contact point -- through the homography)
  before we can produce a valid prediction file at all.
- Usage: `python eval/main.py --prediction_file pred.txt
  --ground_truth_file ground_truth.txt --scene_2_camera_id_file
  scene_name_2_cam_id.json --num_cores N` -- outputs HOTA/DetA/AssA/LocA.
  We'll need our own `scene_2_camera_id.json` scoped to just scene_061's
  4 downloaded cameras (the full one lists all scenes/cameras in the
  dataset, most of which we haven't downloaded).

## Pipeline stages (planned)

1. **Detection + single-camera tracking** -- pretrained YOLO (via
   `ultralytics`) + ByteTrack per camera, producing per-camera tracklets.
   (Note: since ground truth already gives us global object_id per
   camera, we can also validate our own detection/tracking against it
   directly, not just the final cross-camera association step.)
2. **Attribute extraction** -- VLM pass over tracklet crops (reuse
   `../LMM_dive`'s TinyLMM, or a stronger VLM) for clothing/accessory
   description.
3. **Cross-camera identity** -- for this dataset we *have* ground truth
   (object_id), so the interesting work is: how well would a real ReID
   approach recover it? Compare a ReID-embedding-based association
   against the known-correct ground truth -- genuine quantitative eval,
   not guesswork.
4. **Event graph** -- NetworkX (not a full graph DB server -- keep infra
   light). Nodes: entities, events. Edges: PERFORMED, PRECEDES,
   ASSOCIATED_WITH, SAME_ENTITY_AS.
5. **LLM query interface** -- parse natural-language query into a graph
   traversal, execute, generate the answer.
6. **Evaluation** -- no standard benchmark exists for compositional
   natural-language MCT queries, so we hand-curate a small verified set
   from this scene's actual ground truth (same approach as
   `../LMM_dive`'s `check_stage1_grounding.py` / `evaluate.py`).

## Conventions (carried over from ../LMM_dive, still apply)

- Verify dataset/library claims empirically before building on them
  (we've been burned by unverified assumptions before -- e.g. the
  LMM_dive project's COCO-split-mismatch incident).
- Small, runnable increments -- get one scene/3-4 cameras working
  end-to-end before scaling up.
- Quantitative evaluation over vibes -- when a real ground truth exists
  (it does, here), use it.
- Flag tradeoffs and scope decisions rather than deciding silently.

## Current status

- [x] Dataset selected and ethically vetted (nvidia/PhysicalAI-SmartSpaces)
- [x] Ground truth + calibration format decoded
- [x] Downloaded scene_061, cameras 0535-0538 (ground truth + video + calibration)
- [x] Vendored official TrackEval code (`eval_ref/`) -- reproduces the
      README's sample HOTA (49.28%) exactly, so it's trusted as-is
- [x] Detection + single-camera tracking (`scripts/track_single_camera.py`,
      YOLO11 + ByteTrack via ultralytics) + world-coordinate projection
      (`scripts/geometry.py`, homography inverse -- see that file's
      docstring for the empirically-confirmed direction/point convention)
      + prediction-file builder (`scripts/build_pred_file.py`) +
      evaluation wrapper (`scripts/evaluate_tracking.py`)
- [x] Detector tuned on camera_0535 via a small sweep: yolo11n@conf0.3
      (baseline) = 28.06 HOTA / 25.27 DetA; yolo11n@conf0.15 = 27.81 (conf
      alone didn't help); yolo11s@conf0.15 = **33.58 HOTA / 34.15 DetA**
      (model capacity was the real lever); yolo11m@conf0.15 = 32.72 (no
      better than s, not worth the extra cost). Settled on
      **yolo11s @ conf=0.15** as the standing default in
      `track_single_camera.py`.
- [x] Per-camera eval on all 4 cameras with tuned settings -- revealed two
      distinct failure modes, not a single uniform "tracking quality":
      sparse cameras (0535: HOTA 33.6, DetA 34.2; 0536: HOTA 40.9, DetA
      35.3) are detection-recall-limited but associate fine once
      detected; crowded cameras (0537: HOTA 24.2, **DetA 80.1**, **AssA
      7.4**; 0538: HOTA 33.5, DetA 77.5, AssA 14.6) detect well but
      fragment/ID-switch heavily under crowd density. Tried raising
      ByteTrack's `track_buffer` 30->90 (`scripts/bytetrack_tuned.yaml`)
      on camera_0537 to test whether occlusion-duration was the cause --
      it wasn't (HOTA 24.24 -> 24.35, noise-level). This confirms the
      real cause is per-frame ID-swap ambiguity between nearby/
      overlapping people (a matching problem), not track loss over time
      -- fixing it properly needs appearance-based association (e.g.
      BoT-SORT+ReID) rather than a ByteTrack param tweak. Deferred: kept
      as a known, documented per-camera limitation and moved on, since
      the cross-camera ReID step below is itself appearance-based and
      may end up informing a real fix here later.
      Also checked whether a newer detector would trivially fix the
      recall-limited cameras: this `ultralytics` install (8.4.117) does
      ship YOLO26 -- tested yolo26s@conf0.15 on camera_0535, got HOTA
      32.67 vs yolo11s@0.15's 33.58 -- a wash, not an automatic win on
      this synthetic/surveillance distribution (different from COCO,
      which both are pretrained on). Stuck with yolo11s.
- [x] ReID model chosen: `torchreid`'s OSNet x1.0, fine-tuned checkpoint
      from the project's own documented Model Zoo (Market1501, Rank-1
      94.2%/mAP 82.6% per their docs), downloaded to
      `checkpoints/osnet_x1_0_market1501.pth`. Chose this over
      Ultralytics' bundled `yolo26-reid.onnx` specifically because that
      asset's training data isn't publicly documented -- couldn't
      confirm what it was trained on, so couldn't honor the
      Market-1501-family provenance decision below. 512-dim embeddings,
      verified loading + inference on dummy crops.

      **Note on provenance**, since this mirrors the WildTrack decision
      above but with an opposite conclusion, deliberately: this is a
      *pretrained checkpoint*, not a redistribution of Market-1501's
      images -- we never download or touch that dataset's actual photos
      of real people, only reuse someone else's already-completed
      training artifact (same category as using CLIP/YOLO). That's a
      materially different action than what got WildTrack ruled out
      (downloading and building our project's core dataset directly
      from non-consenting real footage). Discussed explicitly with the
      user rather than assumed.
- [x] Cross-camera ReID + evaluated against ground truth object_id
      (`scripts/extract_track_embeddings.py` + `scripts/cross_camera_reid.py`).
      Two real bugs found and fixed along the way, worth remembering:
      (1) crops were passed to torchreid's `FeatureExtractor` in BGR
      (from cv2) but it treats numpy arrays as RGB with no conversion --
      every crop's color signal, the primary ReID cue, was systematically
      wrong. Fixed with an explicit `[:, :, ::-1]` flip in
      `extract_track_embeddings.py`. (2) cross-camera matching originally
      used union-find over every pairwise cosine similarity above a
      threshold -- equivalent to single-linkage clustering, which chains
      badly: at threshold 0.6-0.8 it collapsed ~1200 tracks into a single
      global cluster (HOTA 7.36%, worse than doing nothing). Fixed by
      switching to Hungarian (`scipy.optimize.linear_sum_assignment`)
      one-to-one matching PER CAMERA PAIR, thresholded, with union-find
      only over that sparser, higher-precision match set.
      Result after both fixes: **HOTA 26.90%** at threshold=0.8 (up from
      21.30% with no cross-camera merging at all; AssA 7.00% -> 10.89%,
      the association-quality component this step should move). Full
      threshold sweep (0.5-0.85) was fairly flat around 26-27%, so 0.8
      isn't a sharp optimum, just the best of a shallow curve.
      Diagnosed the remaining gap: the merged output has 748-882 global
      identities vs. only **20** true unique people (confirmed via
      `ground_truth.txt`'s object_id column) across these 4 cameras --
      but only ~280 of those clusters span more than one camera, meaning
      most of the inflation is WITHIN-camera track fragmentation
      (carried over unchanged from the earlier per-camera tracking step,
      especially cameras 0537/0538's 7-15% AssA) rather than a failure of
      cross-camera matching itself. This closes the loop on the earlier
      deferred ByteTrack/appearance-tracking decision with a concrete
      number: single-camera fragmentation, not cross-camera matching, is
      now confirmed as the dominant remaining bottleneck.
      Also checked (per user's explicit direction) domain transfer
      quality of the Market1501-finetuned OSNet on this synthetic
      dataset directly, via same-camera-different-track similarity as a
      clean "different real person" baseline: mean ~0.70, max up to 0.96
      -- weak discrimination, a real domain gap between real photos and
      this dataset's synthetic renders. Noted as a known limitation, not
      fixed (would need domain adaptation or a different backbone).
- [x] Event graph construction (`scripts/build_event_graph.py`, NetworkX):
      entity/sighting/camera nodes, HAS_SIGHTING/IN_CAMERA/PRECEDES edges.
      809 entities, 1160 sightings, 2671 total edges from the merged
      4-camera prediction file.
- [x] Attribute extraction (`scripts/extract_entity_attributes.py`):
      BLIP captions (Salesforce/blip-image-captioning-base) on each
      entity's largest-sighting representative crop -- chosen over
      ../LMM_dive's TinyLMM because that model's own POPE eval showed
      0%-"no"-accuracy hallucination, not reliable enough to build
      queryable descriptions on. Surfaced a real side-finding: some
      captions ("a small dog", "a car") reveal person-class false
      positives from the conf=0.15 detector -- a known, accepted
      tradeoff from the earlier recall-vs-precision tuning decision.
      Found and fixed a real path bug (pred_full.txt's camera_id is an
      unpadded int, directories are zero-padded -- silently produced 0
      crops until fixed).
- [x] LLM query interface (`scripts/graph_tools.py` + `scripts/query_interface.py`):
      Qwen2.5-7B-Instruct with native tool-calling (verified support
      before choosing it), 5 tools (search_by_appearance via
      sentence-transformers semantic search, get_entity_timeline,
      find_entities_in_camera, check_entities_cooccur,
      list_multi_camera_entities). System prompt enforces
      evidence-only answers + explicit "I don't have enough evidence"
      over guessing. graph_tools.py unit-tested standalone against the
      real graph before LLM wiring, per this project's convention.
      Manually verified: correctly distinguished two same-caption but
      different entities rather than conflating them; correctly
      refused to claim cross-camera presence without evidence. Also
      found a real failure mode: one run's final answer directly
      contradicted its own preceding tool result (claimed "no other
      entities" after a tool call had just listed several) -- a
      genuine grounding/consistency gap, not caught by the system
      prompt, documented as a known limitation.
- [x] Compositional query evaluation (`scripts/eval_query_interface.py`):
      6 self-verified test cases (ground truth computed directly from
      the graph, bypassing the LLM). **4/6 (66.7%) passed.** Failures:
      (1) asked the model to count a 273-item tool-result list; it
      answered 783 -- LLMs are unreliable at counting long lists from
      raw text, tools should return precomputed counts, not just raw
      lists, for this kind of query; (2) a genuinely correct, well-
      evidenced "no" answer was graded FAIL because it never used the
      literal word "no" -- an automated-grading strictness artifact,
      not a real model failure.
- [x] Query interface hardened through many rounds of real-usage
      debugging (`--debug`'s full token/tool-call trace was essential
      throughout). Architecture converged on: general, composable tools
      (not one bespoke tool per query shape) that the LLM orchestrates
      itself across as many iterations as needed, with every fact-bearing
      result rendered by a **deterministic Python formatter** and appended
      to the model's answer verbatim -- never retyped, summarized, or
      re-derived by the model. This single pattern was the only reliably
      effective fix across every "the model got a fact wrong" class of bug
      found this session (see "Insights and lessons learned" below).
      Major additions: real spatial proximity (`find_nearby_entities`,
      `find_nearby_entities_by_description`) using per-frame trajectories,
      not just camera+time co-occurrence; `rank_entities_by_interaction_count`
      (full-graph aggregation in code, since per-entity LLM looping is both
      infeasible at ~750 entities and unreliable at counting even smaller
      lists -- one real miscount: 281 actual vs 200 reported);
      `count_nearby_entities` (a lightweight, general "how many interactions
      does X have" primitive); a continuing work-queue for interactor-count
      coverage (drains a few ids per iteration until every discovered
      interactor has been asked about, not just a one-shot first batch);
      `list_low_quality_caption_entities` (see below).
      Found and fixed a real, reproducible model-reliability ceiling:
      batches of 6+ simultaneous tool calls in one turn pushed
      Qwen2.5-7B-Instruct's greedy decoding into generating garbled
      `<tool_call>` tags (a literal stray token replacing the tag),
      silently dropping calls -- and the model's next turn sometimes
      FABRICATED plausible-looking results for exactly the missing ones,
      format-indistinguishable from genuine tool output. Fixed by capping
      simultaneous tool-call batches at 3 (empirically reliable all
      session) plus a structural scrubber
      (`_strip_fabricated_count_claims`) that cross-checks every count
      claim in the model's own prose against ids that actually have a real
      tool result, regardless of whether the fabrication is wrapped in a
      fake `<tool_response>` tag or just bare prose matching the report
      template.
- [x] Caption pipeline quality pass, triggered by a real finding: an entity
      flagged as "non-human" (captioned "a sandwich with a sandwich on it")
      turned out on direct visual inspection to be a real person. This
      recalibrated the whole feature's premise -- most flagged entities are
      believed to be miscaptioned PEOPLE, not genuine non-human objects
      (the person detector fires on the person class, so a false positive
      is far more likely to be a badly-captured real person than an actual
      non-human object appearing in this scene). Renamed
      `list_non_human_entities` -> `list_low_quality_caption_entities`
      throughout, with honest framing in every description/report string.
      Root-caused and fixed two real captioning-pipeline bugs in
      `extract_entity_attributes.py`'s medoid consensus selection:
      degenerate BLIP captions (decoder repetition loops, e.g. "a bald
      bald bald bald...") were eligible to be chosen as consensus over
      coherent, correct captions; and even non-degenerate but WRONG
      captions could still win via pure embedding-distance-to-mean over a
      majority of person-describing captions (the "sandwich" case) -- fixed
      with a degenerate-caption detector plus a majority-vote tiebreak.
      Added two more caption-quality signals, both empirically calibrated
      against the real data rather than guessed: BLIP's own per-token
      generation confidence (an initial guess of 0.5 was checked against
      the actual distribution of 2063 sampled captions -- mean 0.412,
      median 0.409, p90 0.508 -- and found to exclude 88% of ALL captions;
      recalibrated to ~p10, 0.3), and frame-margin exclusion for likely
      cut-off crops (a hard-exclusion first version left 352 of 749
      entities with ZERO extractable crops at all when their whole
      trajectory ran along a frame edge -- fixed to a soft preference with
      fallback). Net effect: 58 -> 30 flagged entities.
- [x] Regression test suite added (`tests/`) -- see "Tests" below.

## Additional data sources: WildTrack, POM, EPFL-RLC

Beyond the primary nvidia/PhysicalAI-SmartSpaces pipeline (scene_061),
three more EPFL CVLab multi-camera datasets were added, each independent
(no cross-scene merging -- the graph schema has no notion of "which world
this happened in," and no query needs cross-scene reasoning). Each keeps
its own `pred_full.txt`/`event_graph_with_attrs.gpickle`, run with
`query_interface.py --graph <path>` explicit each time (scene_061's
default is untouched).

**Ethics, stated plainly and deliberately** (this section exists so the
decision is visible, not silent, mirroring how the original WildTrack
rejection above was documented): two of these three sources are the SAME
category of ethical concern that got WildTrack originally rejected --
real people, recorded via camera, without documented meaningful informed
consent. This was raised directly and explicitly in conversation before
any of the three was touched, not discovered after the fact or glossed
over.

- **WildTrack** (`data/wildtrack/cam1.mp4`..`cam7.mp4`, 7 cameras,
  1920x1080 ~60fps ~125,850 frames each, no calibration or ground truth
  available -- videos only): this is the EXACT dataset rejected earlier in
  this file (real students, no meaningful informed consent, exposing.ai's
  DukeMTMC parallel). The user was told this explicitly, in these terms,
  before proceeding, and chose to use it anyway -- **a deliberate,
  informed reversal of the earlier decision, not a default or an
  oversight.** Implementation deferred (Phase 2 of the approved plan,
  needs its own approval before starting) -- no calibration means no
  world-coordinate reasoning is possible for this source regardless.
- **EPFL-RLC** (`data/EPFL-RLC/`: JPEG frame sequences, not video; Tsai
  calibration XMLs; grid-based ground truth): real people recorded in the
  EPFL Rolex Learning Center (a semi-public space), with **no consent or
  ethics statement anywhere on the dataset's page** -- same concern class
  as WildTrack, independently confirmed by fetching and reading the page
  directly (not assumed). The user was told this explicitly and chose to
  proceed anyway, **the same kind of deliberate override as WildTrack,
  not a default.** Implementation deferred (Phase 3 of the approved plan).
- **POM terrace1** (`data/POM/terrace1/`: 4 cameras, 360x288, 5010 frames
  each, from EPFL CVLab's "Multi-Camera Pedestrians" / Probabilistic
  Occupancy Map dataset): **no override needed here** -- the dataset's own
  page states "All pedestrians on the sequences are members of our
  laboratory, so there is no privacy issue," genuine documented consent,
  a materially different situation from the other two. This is the one
  fully implemented so far (below).

### POM terrace1 -- implemented end-to-end

New file `scripts/adapt_pom_calibration.py` converts POM's native
calibration into the existing `calibration.json` shape (confirmed by grep
that nothing downstream reads anything except its `"homography matrix"`
field, so `geometry.py`/`build_pred_file.py`/etc. needed zero changes).

**Homography direction -- verified empirically, not assumed, and a real
mistake caught along the way**: POM ships calibration two ways -- a
directly-given ground-plane homography (`calibration-terrace.txt`) and an
independent Tsai camera model (`terrace-tsai.zip`) for the same 4 cameras.
The Tsai-based cross-derivation (reconstructing `P=K[R|t]`, dropping the Z
column) was tried first as the primary check, but its best-case residual
against the given homography was 90-300% across every camera -- far too
large to trust, most likely from a wrong assumption reconstructing this
specific toolkit's exact rotation/unit convention, which isn't documented
anywhere found. Fell back to the plan's secondary method: ran real YOLO
detections on all 4 cameras, mapped them through both candidate uses of
the given homography, and checked which one made DIFFERENT cameras'
detections of the same real people converge on the same world coordinates
(the same physical terrace, observed from 4 angles, should show exactly
that under a correct calibration). First attempt at scoring this used
mean pairwise distance across all points, which gave the WRONG answer --
it conflates "different real people" (expected to be far apart
regardless of calibration correctness) with "the same person seen by two
cameras" (should collapse to ~0 under a correct calibration), and is
dominated by the former. Fixed by scoring the mean nearest-cross-camera-
detection distance instead (11.6 for the winning case vs. 58.9 for the
losing one) -- isolates the actual signal. Verified further downstream:
per-camera world-coordinate medians cluster tightly together (~160-182,
~219-232) across all 4 cameras once the correct direction is used.

**A real, unrelated performance bug found and fixed while building
this**: `extract_entity_attributes.py`'s crop extraction used to
`cap.set(CAP_PROP_POS_FRAMES, ...)`-seek to each target frame
individually. Measured directly on POM's video files (encoded with the
old `IV50` codec, from 2008): a single random seek took ~0.83s vs.
~0.0016s for a sequential read -- ~500x slower, turning what should have
been a multi-second job into a multi-hour one (caught mid-run: 25+
minutes of CPU time with 0% GPU utilization, i.e. still stuck before BLIP
inference even started). Fixed by restructuring crop extraction into two
phases per camera: resolve every target's exact frame/bbox first (pure
dict lookups, no video I/O), then do ONE sequential pass over the video
grabbing every needed frame as it's encountered. This applies to every
source, not just POM -- scene_061's already-fast-seeking H.264/MP4 files
aren't disadvantaged by it either, so it's a strict improvement, not a
POM-specific special case.

**gt_terrace1.txt -- partially decoded, evaluation deferred**: 5010 rows
(one per frame, matches the video's real frame count) x 9 tab-separated
columns, values `-1`/`-2`/occasional positive integers at roughly regular
spacing. Consistent with "9 person-slots, a positive value is a grid-
cell/position index" (the Fleuret/POM toolkit's known style), not raw
world coordinates -- still one indirection away from usable per-frame
(person, world_x, world_y) tuples. Not fully decoded; quantitative HOTA
evaluation for this scene is a documented, deliberate scope gap, not a
silent omission. EPFL-RLC's `mv_examples/*.json` turned out to be a
single-frame(ish) positive/negative ROI-classifier training set (the
original POM detector's own training data), not a trajectory ground
truth at all -- evaluation there isn't just deferred, it's **not planned**,
since the resource isn't shaped for it regardless of decoding effort.

Result: 934 entities, 4 cameras, 129 identities spanning >1 camera (out
of 1163 raw per-camera tracks -- the same within-camera-fragmentation-
dominant pattern already documented for scene_061). Manually verified via
`query_interface.py --graph data/POM/terrace1/event_graph_with_attrs.gpickle`:
multi-camera entity counts and per-entity camera lists come back sane.

## Fixing captions, identity matching, and adding actions

Triggered by real usage of POM: captions hallucinated activity/scene
context (the "tennis"/"sandwich" precedents), identity matching was badly
fragmented (934 entities for "up to 7 people"), and action description
was a stated long-term goal. Plan: fix captions and identity matching, add
actions, validated on POM first, then rolled out to scene_061 (real
ground truth) and EPFL-RLC. Phases below track the approved plan.

### Phase 0 -- captioning model bake-off

Real candidates (SAM3 and DeepSeek-OCR/Donut excluded up front -- not
captioning models; SAM3 belongs to a separate, deliberately-deferred
"detect arbitrary movable objects" decision) compared on instruction-
following (the structural fix for hallucinated activity claims -- BLIP-
base's `text=` argument is a caption PREFIX to continue writing from, not
a directive it can obey, which was the actual mechanism behind the
tennis/sandwich hallucinations, not just "BLIP is weak") and multi-frame/
video support (needed for action description later, so a model lacking
it would mean two model families instead of one). Shortlist: Qwen2.5-VL-
3B, LLaVA-OneVision, Molmo, BLIP-2/InstructBLIP (cheap control).

Molmo was blocked before it could run: its `trust_remote_code` modeling
file statically imports `tensorflow` (unused at runtime, but
`transformers`' import-check is static) -- installing full TensorFlow
into `mct-env` just to satisfy that check was rejected as inconsistent
with this project's slim-env convention. Decided from the 3 remaining
candidates instead of accepting the bloat.

Ran all 3 on the 18 real POM crops with the lowest `caption_agreement`
(the already-known-problematic ones), both unconstrained and with a
constrained appearance-only prompt. ALL THREE fully eliminated activity/
scene hallucination under the constrained prompt (18/18 clean, vs. their
own unconstrained outputs on the same crops, which invented specific
scenes/activities every time -- "airport", "train station", "skateboard").
The real differentiator: one crop (an 8x112px sliver, almost certainly a
person-class false-positive detection) got a confidently-invented "red
shirt with a white collar" (Qwen2.5-VL-3B) or "red jacket and a black
hat" (LLaVA-OneVision) despite an explicit "if unclear, say so" hedge in
the prompt -- neither model used it. BLIP-2 returned an empty string
instead (not principled uncertainty-awareness, likely just BLIP-2's
typical behavior on a low-information image) -- so strictly, NONE of the
three reliably self-report uncertainty via prompting alone. This
reproduces this project's most-validated lesson (prompt-only fixes are
unreliable) in a new spot: the real fix is a deterministic crop-quality
gate BEFORE any VLM call (see Phase 1), not picking whichever model
happens to hedge best.

**Decision: Qwen2.5-VL-3B**, for both captioning (Phase 1) and action
description (Phase 4 -- a unified model family) -- the only one of the
three with native multi-frame/video support, and its constrained-prompt
descriptions were also the most detailed/useful once color is handled
separately (below). BLIP-2 remains useful evidence that model SIZE
wasn't the deciding factor -- instruction-following + multi-frame
support was.

### Phase 1 -- captioning fix (POM)

`scripts/extract_entity_attributes.py` rewritten around Qwen2.5-VL-3B
with a constrained prompt (clothing type + accessories only, no color,
no activity -- validated in the Phase 0 bake-off) plus two deterministic
fixes layered on top, both added because the bake-off showed prompting
alone isn't reliable enough:

- **`scripts/crop_quality.py`** -- a blur/size gate that skips the VLM
  entirely for crops too degenerate to read, replacing them with a fixed
  `"appearance unclear (low-quality crop)"` caption instead of risking a
  confident guess. Real miscalibration caught and fixed during this
  phase, not shipped blind: the first version calibrated thresholds
  against a RANDOM (camera, frame) sample from `pred_full.txt`, but
  `extract_entity_attributes.py` actually samples crops spread across
  each ENTITY's own trajectory -- a measurably different, smaller-skewed
  population (real area p5=364 vs. the random-frame sample's p5=1135).
  Applied blindly, this flagged 41% of all 934 real entities as
  "unclear" -- caught by comparing the actual full-run output rate
  against the calibration's expected ~5%. Re-diagnosed by pulling real
  borderline crops and visually inspecting them (same method that caught
  the original "sandwich" bug): a legitimate 16x185px crop (narrow only
  because the person was captured in a tight upright bbox) was being
  rejected by a WIDTH-only rule despite being perfectly legible, while
  the bake-off's actual failure case (the 8x112px sliver) had a
  deceptively large AREA (896px) that made geometric size alone
  insufficient to catch it -- that failure mode turned out to be about
  crop CONTENT (a solid-color blob with no structure), not size, and no
  purely geometric rule catches it perfectly. Recalibrated to
  area-only (>=400px, from direct visual inspection of real
  confirmed-bad ~266-290px examples vs. a confirmed-legible ~624px one)
  plus the original conservative Laplacian-variance blur threshold,
  dropping the width-only rule entirely. Result: 8.2% of POM's 934
  entities flagged unclear (a plausible, honest rate) instead of 41% (a
  bug), and `caption_agreement < 0.6` dropped from 105/934 (11.2%, the
  original BLIP baseline) to 10/934 (1.07%).
- **`scripts/color_utils.py`** -- RGB/Lab dominant-color sampling
  (k-means over a garment region -- upper-body/lower-body bbox-fraction
  slices -- in Lab space, largest cluster, nearest-named-color match)
  replaces the VLM's own color guessing entirely. Also motivated by the
  bake-off: an explicit "do NOT mention color" instruction was NOT
  reliably honored by any of the 3 real candidates either, so
  `strip_color_words()` deterministically removes color tokens
  (including hyphenated compounds like "dark-colored", a real gap in
  the first version) from the VLM's garment-type text before splicing
  in the RGB-sampled color -- color in the final `appearance_caption`
  always comes from actual pixels, never the captioning model, no
  matter how the prompt is phrased.

Per-token generation confidence (`CAPTION_CONFIDENCE_THRESHOLD`, carried
over from BLIP's calibrated 0.3) was CHECKED against Qwen2.5-VL-3B's real
distribution (193+ real captions: min=0.921, p10=0.975, mean=0.992) and
found not to transfer as a useful signal at all -- the model's greedy
decoding on this short, templated, low-entropy task is almost always
sharply peaked, unlike BLIP's more variable open-ended captioning, so
there's no meaningful low-confidence tail to gate on. Left in place as an
effectively-inert safety net rather than replaced with an arbitrary cutoff
inside the observed 0.92-1.0 band. The degenerate-caption check and
cross-crop `caption_agreement` remain the two signals that actually work
for this model.

`query_interface.py`'s system-prompt paragraph warning about the old
"tennis"-style activity hallucination was rewritten -- it described a
failure mode that's now structurally prevented (`appearance_caption` is
appearance-only by construction), and referenced a specific
caption_agreement rate that was scene/pipeline-version-specific, not a
durable fact about the system.

### Phase 2 -- within-camera identity stitching (POM)

`scripts/cross_camera_reid.py` extended with a WITHIN-camera matching
pass that runs FIRST, before the existing cross-camera pass -- this
script previously only ever matched tracks ACROSS cameras, so nothing
had ever stitched a single camera's own ByteTrack ID-swap fragments back
together, despite that being the already-documented dominant cause of
entity over-counting (20 true identities vs. 809 predicted on
scene_061). Same conservative mechanism as the existing cross-camera
matcher (Hungarian one-to-one per pair, thresholded, union-find only
over the matched set) extended to a same-camera pass rather than
reverted toward global thresholded matching -- the approach that
previously caused a catastrophic 1200-tracks-into-one-cluster collapse.

The confirmed highest-risk detail from planning -- reusing the cross-
camera plausibility rule unchanged for same-camera pairs would let two
simultaneously-visible, genuinely different people within
`OVERLAP_DISTANCE_TOLERANCE_M` (3m) of each other merge -- was fixed as
planned: `plausibility_matrix` takes a `same_camera` argument, and when
true, ANY temporal overlap is a hard veto (no distance exception), vs.
cross-camera's "only overlapping AND far apart" rule (two cameras CAN
legitimately see the same person from different angles at once).

`extract_track_embeddings.py` also gained the SAME crop-quality gate
from Phase 1 (`crop_quality.is_low_quality_crop`), applied before
computing each track's ReID embedding -- motivated by a real finding
during a same-camera-threshold sweep on `camera_0000`: a 3-way "merge"
group turned out to be several degenerate/uninformative crops (the same
kind of unreadable sliver Phase 1's gate already catches for captioning)
clustering together in embedding space on shared noise, not genuine
appearance similarity. Tracks whose every sampled crop fails the gate
are excluded from `track_embeddings.npz` entirely (no embedding) rather
than embedding pure noise -- `build_pred_file.py` already handled this
gracefully (skips detections with no `global_id_map` entry, an existing
"no valid crops" path), so no changes needed there.

Two real, sequential bugs in the group-consolidation step, found via a
direct before/after comparison rather than shipped blind (cross-camera
spanning-identity count collapsed from 121, no within-camera stitching,
to 13 with the first version -- caught by comparing full runs with vs.
without `--skip_same_camera` on identical crop-quality-filtered inputs,
isolating the cause to consolidation itself):
- **Embedding pooling**: the first version MEAN-pooled every group
  member's embedding into one consolidated vector. Averaging multiple
  crops of genuinely the same person can land in a "blended" region of
  embedding space that's LESS similar to a correct cross-camera match
  than any single strong fragment would be -- and averaging a WRONGLY-
  merged group (confirmed to happen at a real, if modest, rate -- see
  below) makes this worse, blending two different people's signal into a
  third that matches neither well. Fixed by using the group's LARGEST
  member's (longest frame span, a proxy for most detections) embedding
  as-is, no averaging. This alone barely moved the number (13 -> 12) --
  a real improvement to keep, but NOT the dominant cause.
- **Frame-range widening (the actual dominant cause)**: also
  consolidated each group's first/last frame to min/max across ALL
  members, which artificially WIDENS its temporal footprint -- e.g. two
  fragments at frames 1000-1200 and 5000-5200 become one group spanning
  [1000, 5200]. `plausibility_matrix`'s cross-camera overlap branch then
  spuriously fires against unrelated candidate tracks anywhere in that
  wide window; and even when not hard-vetoed, its `required_speed =
  dist/gap_sec` formula floors `gap_sec` at 1 frame for ANY overlapping
  pair, collapsing plausibility to ~0 for all but near-zero distances --
  so widening the window silently converted legitimate "sequential,
  walkable gap" cross-camera candidates into spurious "overlapping,
  implausible" ones before Hungarian ever saw them. Fixed by using the
  same representative member's own TIGHT (unwidened) frame range instead
  of the group's envelope. This recovered spanning identities to 71 (of
  125 cross-camera merges) -- lower than the original 121, but expected
  and correct, not a regression: before within-camera stitching, one
  real cross-camera person fragmented into several same-camera pieces
  could generate multiple separate "spanning" counts (an artifact of the
  fragmentation this phase exists to fix); after stitching, that's
  correctly one spanning identity, counted once.

Same-camera threshold chosen (0.75) via a small sweep on `camera_0000`
alone (317 raw tracks; 0.5 collapsed to 166 groups, 0.9 only to 285) plus
direct visual spot-checks against real video frames (the same method
that caught the original "sandwich" caption bug) on a random sample of 5
merged pairs at threshold=0.75: 4/5 were clearly the same person by
direct inspection, 1/5 was a likely wrong merge (different hair/build/
colors) -- an honestly-reported ~80% spot-check precision on a tiny
sample, not a rigorously validated rate, consistent with this dataset's
already-documented OSNet domain-gap limitation (many outfits are simply
generic dark clothing, which weakens appearance discrimination
regardless of threshold).

Result after both bug fixes, run on all 4 POM cameras: 1123 raw tracks
(down slightly from 1163 due to the new crop-quality filter) -> 712
within-camera groups -> 589 final global identities (a ~37% reduction
from Phase 1's 934, entirely from collapsing within-camera fragmentation
-- POM's own stated "up to 7 people" is still far below 589, confirming
within-camera fragmentation remains the dominant, not fully solved,
source of over-counting, consistent with scene_061's much larger
20-vs-809 gap). `caption_agreement < 0.6` dropped further to 5/589
(0.85%, down from Phase 1's 10/934). Re-ran Phase 1's full attribute
extraction on the new graph (entity ids changed after re-merging) and
verified live via `query_interface.py`.

A genuinely positive finding surfaced while investigating Phase 3 below:
the single largest merged group (22 raw tracks, spanning all 4 cameras)
turned out, on direct visual inspection of 6 samples spread across it, to
be a REAL correct match -- one person in a distinctive black top + light
blue jeans combination (unusual against this population's mostly-dark
clothing), recurring across dozens of fragments over the ~5000-frame
video. A large merged group isn't automatically suspicious -- it can be
a real person who spent a lot of time in view, correctly reassembled
from many fragments.

### Phase 3 -- caption similarity as a third matching signal (POM) -- implemented, evaluated, disabled by default

New `scripts/extract_track_captions.py`, mirroring `extract_track_
embeddings.py`'s two-phase sequential-read structure: captions LOCAL
(pre-merge) tracks with the same Qwen2.5-VL-3B constrained prompt as
Phase 1, since captions previously only existed AFTER the graph was
built -- too late for a signal that needs to inform merging itself.
`cross_camera_reid.py`'s `match_pairs` extended to optionally combine
`w_appearance * appearance_sim + w_caption * caption_sim` (additive, not
another hard multiplicative gate, since captions are noisier/more
generic) before multiplying by plausibility, falling back to appearance-
only for any pair missing a caption embedding.

**Evaluated honestly, found to net-hurt on POM, and disabled by default
rather than shipped on the strength of the plan's a-priori reasoning
alone.** At the plan's starting weight (`w_caption=0.2`), the largest
merged group grew from 22 raw tracks (appearance-only) to 33 -- and
critically, those 33 draw from 11 DIFFERENT appearance-only groups, not
just a few extra fragments of the same recurring blue-jeans person.
Direct visual inspection of 3 of those contributing groups showed clearly
different people (a brown-jacket person, a maroon-short-sleeve-top
person, a person with a different build) incorrectly pulled together.
Sweeping `w_caption` down (0.1, 0.05, 0.02) did NOT cleanly fix this --
group sizes stayed noisy and elevated (39, 38, 24) even at very low
weights, converging toward the appearance-only baseline only as the
caption contribution became negligible. Root cause, not just a threshold
problem: this population's captions are too GENERIC to serve as a
reliable identity signal -- most people wear similarly nondescript dark
clothing (already documented above), so the constrained prompt's limited
vocabulary (top/bottom color + garment type) produces near-identical
captions for many DIFFERENT real people, and any non-trivial caption
weight lets that false similarity bridge pairs the appearance signal
alone had correctly kept separate -- the same transitive union-find
chaining risk this whole matching design was built to avoid, reintroduced
through a second, less-discriminative signal.

Kept in the codebase as an opt-in feature (`--w_caption`, default 0.2,
only takes effect where `track_captions.npz` files exist) rather than
deleted -- the underlying idea (match by appearance + wearing + position,
per user direction) is sound in principle, and a scene with more visually
diverse clothing than this one might get real value from it. POM's own
production `global_id_map.json`/graph stay on Phase 2's appearance-only
result; this finding should be re-checked (not assumed either way) before
enabling the caption signal on scene_061 or EPFL-RLC in Phase 6.

### Phase 4 -- action description (POM) -- the original headline ask

New `scripts/extract_entity_actions.py`, structurally mirroring
`extract_entity_attributes.py` but for BEHAVIOR: 1-2 short CLIPS (12
consecutive frames, ~0.4s @ 30fps) per entity instead of static crops,
fed to Qwen2.5-VL-3B's native video input (verified directly before
building the pipeline: a real 12-frame walking clip correctly produced
"Walking", confirming the mechanism works before investing in the full
build). Constrained prompt is the inverse of Phase 1's -- describe
ONLY behavior, explicitly forbidding appearance/clothing/color claims,
with the same "say 'walking'/'standing still' rather than inventing an
interaction" hedge validated in spirit by the Phase 0 bake-off. New
`action_description` + `action_agreement` fields (same medoid-consensus-
over-multiple-clips computation as `appearance_caption`/
`caption_agreement`), plus `GraphTools.search_by_action()` /
`get_entity_action()` and matching `query_interface.py` tools, built
from the exact `list_low_quality_caption_entities`/
`list_human_captioned_entities` template (shared logic, explicit tool-
schema cross-referencing to prevent the tool-mixup bug already caught
once this session).

**A real bug found and fixed before the first successful run**: a
person's bbox shifts slightly frame to frame (walking, turning), so
per-frame crops within one clip aren't identically sized -- `qwen_vl_
utils`' `process_vision_info` stacks video frames into one tensor and
requires matching dimensions, and crashed on the first real clip. Fixed
by resizing every frame in a clip to its first frame's size before
building the video input (a mild resize within one ~0.4s clip, not a
distorting one).

**Real, honestly-reported failure modes found via direct visual
inspection against real footage** (same discipline that caught the
original "tennis"/"sandwich" caption bugs), NOT fixed this phase --
documented as known limitations rather than papered over:
- **The appearance-forbidding prompt is not fully reliable**, same
  pattern as Phase 1's color-suppression finding: one real output read
  "The person in the black jacket appears to be walking forward" --
  clothing leaked through despite the explicit instruction. No
  deterministic scrub was added for this (unlike Phase 1's color
  splicing) since general clothing vocabulary is far larger and fuzzier
  than the small enumerable color-word list `strip_color_words` handles
  -- flagged as a real, unaddressed gap, not silently accepted.
- **Confident over-interpretation of ambiguous small motions**: one
  clip produced "The person appears to be holding their head in their
  hands, possibly indicating discomfort or distress" from what was
  likely just a small head movement -- an emotional-state inference the
  constrained prompt never asked for and the model shouldn't be
  confident enough to offer.
- **A confirmed false claim, verified by direct frame inspection**:
  asked live "is anyone carrying something or sitting down?", the
  system answered with entity 341 as a confident, 1.0-agreement
  "sitting" match. Pulling three real frames from the exact clip used
  (frames 1471, 1477, 1482 of camera_0002) showed the person clearly
  WALKING/standing in all three -- no chair, no bent knees, full
  standing posture. This is a genuine action-description hallucination,
  not a labeling ambiguity, and the ONLY one of the spot-checked claims
  that turned out wrong -- the more common "walking"/"standing still"
  outputs held up under the same direct-inspection check.
- Per-token action-description confidence (same `ACTION_CONFIDENCE_
  THRESHOLD=0.3` starting value as Phase 1's caption threshold) checked
  against a real run and found, again, not to transfer as a useful
  signal: mean=0.995, p10=1.000, max=1.000 across all real POM action
  descriptions -- left in as an inert safety net for the same reasons
  already documented for `CAPTION_CONFIDENCE_THRESHOLD`.

Result on POM: 589 entities described, `action_agreement < 0.6` for
102/589 (17.3%) -- meaningfully higher than appearance captions' 0.85%,
but not necessarily a quality problem in the same sense: unlike
appearance (which should be stable across an entity's crops),
BEHAVIOR can genuinely differ between two different clips of the same
entity (e.g. walking in one, standing still in another) without either
clip being wrong -- low agreement here is a noisier, less directly
comparable signal than `caption_agreement`, not a like-for-like
regression.

### Phase 6a -- roll out to scene_061 -- the real ground-truth check

Full pipeline re-run on scene_061's 4 cameras (0535-0538): re-extracted
track embeddings with the new crop-quality gate, within-camera stitching
(`--same_camera_threshold 0.75`, reusing POM's value) + cross-camera
matching (`--threshold 0.8`, the pre-existing scene_061 default) ->
rebuilt `pred_full.txt`/event graph -> Qwen2.5-VL-3B captioning ->
action description. Caption-similarity (Phase 3) stayed off by default,
consistent with the POM finding.

**This is the strongest evaluation this whole effort has had access to,
and it confirms the within-camera stitching genuinely helps, not just
reduces a count**: `evaluate_tracking.py` against real ground truth gave
**HOTA 27.02% / AssA 11.27% / DetA 64.94%**, both HOTA and AssA UP from
the documented pre-this-session baseline (HOTA 26.90%, AssA 10.89%, from
cross-camera-only ReID). Entity count dropped from 809 (documented
baseline) to 364 raw-track-count-adjusted total (866 raw tracks -> 447
within-camera groups -> 364 final identities), against a true GT_IDs of
20 -- still far from ground truth (confirming, yet again, that within-
camera fragmentation is a large, not-yet-fully-solved problem, most
severely on the two crowded cameras: camera_0537 alone had 644 raw
tracks pre-merge, vs. camera_0535/0536's 30/16), but a real, measured
step in the right direction rather than an assumed one -- exactly the
check Decision 5 called for (a lower entity count alone isn't sufficient
evidence, since wrongly-merged entities would also lower it while making
HOTA worse; here HOTA and AssA both improved, so the reduction reflects
genuine fragmentation being fixed, not silent over-merging).

Captioning: 21/364 entities (5.77%) `caption_agreement < 0.6` -- a real,
different population from POM (indoor/outdoor mixed, different crowd
density) but in the same low-single-digit range Phase 1 established.
Action description: 115/364 (31.6%) `action_agreement < 0.6`, higher
than POM's 17.3% -- consistent with the already-documented "behavior
varies more than appearance, not necessarily a quality problem" caveat,
though also plausibly reflecting scene_061's crowded cameras producing
messier clips than POM's more sparsely-populated terrace. Verified live
via `query_interface.py` (combined multi-camera-entity-count +
walking-search query, both tool families working together on the same
graph).

### Phase 6b -- roll out to EPFL-RLC

Same full pipeline re-run on RLC's 3 cameras. No ground truth here (per
the earlier-documented decision -- RLC's own `gt_terrace1.txt`-equivalent
resource isn't shaped for evaluation), so this phase relies on the
entity-count sanity check plus live spot-checks, not a HOTA gate.

Raw tracks: 3626 (previously documented) -> 2944 after the new crop-
quality filter excluded some all-degenerate tracks from getting an
embedding at all (same mechanism as POM/scene_061). Within-camera
stitching (2914 merges) -> 1429 groups -> cross-camera matching (517
merges) -> **912 final entities, down from the previously-documented
2301** (a ~60% reduction) -- directionally consistent with both other
scenes' within-camera-stitching wins, though RLC still has no ground
truth to confirm this is genuine fragmentation-fixing rather than partial
over-merging the way scene_061's HOTA/AssA numbers could confirm.
`caption_agreement < 0.6`: 54/912 (5.9%) -- in the same low-single-digit
range as POM (1.07%, Phase 2's canonical result) and scene_061 (5.77%),
i.e. captioning quality transferred cleanly to this source.

**Action description transferred less cleanly**: 405/912 (44.4%)
`action_agreement < 0.6` -- notably higher than both POM (17.3%) and
scene_061 (31.6%), a real, honestly-reported gap rather than an assumed-
fine number. Plausible (not confirmed) explanation: RLC's frames are
natively only 480x270 (vs. scene_061/POM's 1920x1080-derived sources),
its own documented lowest-resolution source in this project -- a clip
built from already-small crops has less visual information for the model
to reliably describe consistent behavior across the several fragments a
single entity might contribute frames from. Not investigated further
this session (would need direct frame-by-frame inspection of several
RLC action clips, similar to Phase 4's POM investigation, to confirm vs.
rule out); flagged as a known, scene-dependent limitation rather than
silently accepted as equivalent to the other two scenes.

Verified live via `query_interface.py` (multi-camera-entity-count query
returned a plausible result on the rebuilt graph).

**All three scenes (scene_061, POM terrace1, EPFL-RLC) now run the same
captions/matching/actions pipeline** -- Qwen2.5-VL-3B constrained
captioning with RGB-sampled color, within-camera + cross-camera identity
stitching, and action description, each independently rebuilt and
verified (scene_061 via real HOTA/AssA against ground truth; POM and RLC
via entity-count sanity checks, direct visual spot-checks, and live
queries). Caption-similarity (Phase 3) stays off by default everywhere,
per the POM finding that it isn't a reliable signal on any of this
project's populations checked so far.

### Phase 7 -- post-rollout deep dive: is 809->364 (or 934->589) actually close enough?

Triggered by direct user pushback after Phase 6: POM terrace1's "up to 7
people" vs. 589 predicted entities is not a rounding error, and comparing
which within-camera-fragmentation FIX produces fewer raw tracks (317 vs.
369 vs. 357 in an earlier BoT-SORT sweep) is beside the point if ALL of
them are off by an order of magnitude. This phase is a real, evidence-
driven investigation into WHY, not another round of threshold tuning.

**BoT-SORT+ReID (single-camera tracker replacement) -- tried, real bugs
fixed along the way, ultimately NOT adopted (real HOTA test, not just
track-count heuristics):**
- Exported the project's own already-vetted OSNet-Market1501 checkpoint
  to ONNX (`scripts/export_osnet_onnx.py`) rather than use ultralytics'
  bundled `yolo26-reid.onnx` asset, for the same provenance reasons
  already documented for the original ReID model choice (verified the
  export is faithful: cosine similarity 0.99998 against the original
  PyTorch model on a real crop).
- Found and fixed two real bugs before the tracker would even run
  correctly: ultralytics' BoT-SORT ReID hook (1) hard-codes SQUARE input
  resizing, incompatible with OSNet's trained 256x128 rectangular aspect
  (would silently distort every embedding, not crash); (2) only applies
  `/255` scaling, no ImageNet mean/std normalization, which this
  project's OSNet export needs. Fixed via a local runtime monkeypatch
  (`scripts/reid_rect_patch.py`) rather than editing the installed
  package, preserving upgrade-safety.
- Found a THIRD real bug via direct `nvidia-smi` monitoring, not assumed
  fixed just because it ran without error: ONNX Runtime's
  `CUDAExecutionProvider` silently failed to actually engage the GPU on
  this machine (~3-4% utilization throughout a run, a real onnxruntime-
  gpu/CUDA-toolkit version mismatch) despite being "selected" with no
  error. Routed around entirely -- `TorchReIDEncoder` in the same patch
  file calls this project's own already-fast, already-correct torchreid/
  PyTorch/CUDA `FeatureExtractor` directly, bypassing ONNX Runtime.
- Extensive empirical testing on POM camera_0000, all NEGATIVE: default
  BoT-SORT+ReID (369 tracks, worse than ByteTrack's 317); disabling
  `gmc_method` (357, since `sparseOptFlow` motion compensation is
  designed for moving cameras, not this dataset's static ones); a full
  `appearance_thresh` sweep (0.5/0.6/0.7/0.9, all 357-360, and 0.5-0.7
  tied exactly -- meaning appearance matching wasn't even the binding
  constraint in that range). Nothing beat ByteTrack.
- **Decisive test, at the user's explicit direction**: real HOTA
  evaluation (not track counts) on scene_061's camera_0537 -- the
  crowded, AssA-limited camera (documented ByteTrack baseline: HOTA
  24.2%, DetA 80.1%, AssA 7.4%) BoT-SORT+ReID was specifically
  hypothesized to help most. Result: **HOTA 24.86%, DetA 80.06%, AssA
  7.74%** -- essentially a wash, noise-level differences in both
  directions, not a real improvement. **Not adopted.** Kept as
  documented, working infrastructure (`scripts/botsort_reid.yaml`,
  `scripts/reid_rect_patch.py`, `scripts/export_osnet_onnx.py`) in case a
  future, more discriminative ReID model changes this conclusion --
  ByteTrack remains the shipped single-camera tracker.

**cross_camera_reid.py -- four real bugs found via direct visual
evidence, three fixed and kept, one found and deliberately reverted:**

1. **Boundary-touch overlap veto** (fixed): the same-camera hard veto's
   overlap check (`first_b <= last_a`) treats a track ending at frame X
   and another starting at frame X as "simultaneous," when it's actually
   the single most common ID-switch signature -- confirmed concretely
   (POM camera_0000 tracks 419/488: literally the same detection at the
   same frame, split by the tracker). Fixed by requiring overlap
   DURATION to exceed a small grace window (`SAME_CAMERA_OVERLAP_GRACE_
   FRAMES=2`) before triggering the veto -- a genuine two-people-at-once
   case spans many frames of real shared visibility, not a boundary
   touch, so this doesn't risk conflating them.
2. **One-shot Hungarian under-merging multi-way chains** (fixed): a
   single Hungarian round enforces at most one match per track, so a
   person fragmented into 3+ pieces can only pair up floor(n/2) of them
   per round even with every pairwise similarity above threshold --
   confirmed concretely (419 matched to 749, 488 matched to 928 in round
   1, even though 419-488 themselves scored 0.765 -- Hungarian's GLOBAL
   optimum simply preferred other pairs). Fixed with an ITERATIVE loop:
   repeat consolidate-then-match until no new merges, letting chains
   resolve over several rounds via "pairs matching pairs." Threshold
   ESCALATES each round (`--same_camera_threshold_increment`, default
   0.03) since a consolidated representative gets noisier the more
   merging it's absorbed -- flat-threshold iteration was tried first and
   confirmed worse via spot-check (2/6 wrong in the largest resulting
   group vs. 1/6 with escalation).
3. **Unreliable world-position math for tight handoffs** (fixed): the
   existing world-position signal (a track's MEDIAN position across its
   whole lifetime) sits hundreds of world-units from a track's actual
   endpoint for any track that moved substantially -- confirmed
   concretely (a visually-confirmed real handoff pair showed 899 world-
   units apart by median position, vs. ~13-35 by PIXEL-space endpoint
   position). `extract_track_embeddings.py` now also saves first/last-
   frame PIXEL positions (not just world positions); a new "clean
   handoff" signal (tight time gap AND tight pixel distance) can OVERRIDE
   a noisy appearance score -- confirmed necessary, not cosmetic: several
   real same-person handoffs had appearance similarity as low as 0.59
   (likely motion blur/tiny-crop noise), which no reasonable appearance-
   only threshold could accept without also accepting many false
   positives elsewhere. Two sub-bugs caught and fixed while building
   this: a linear-decay-from-zero confidence formula gave almost no
   credit to cases well inside the calibrated range (fixed to a full-
   confidence-then-decay shape); the handoff score was originally still
   multiplied by the same unreliable world-position-based plausibility,
   silently zeroing it back out (fixed by applying it AFTER, as an
   independent path to a match, not gated by the same unreliable signal
   it was built to route around).
4. **Group-pooled similarity** (found, tried, REVERTED -- a real
   regression, not shipped): the three fixes above still left many small
   groups matching some OTHER member of a large (60-83-member) group far
   better (0.88-0.95) than that group's single chosen representative
   (0.6-0.8) -- confirmed by direct comparison, a genuine "single
   representative under-represents a large group" bug. Fixing it (max,
   then top-3-mean, pairwise similarity across ALL members of both
   groups) made things WORSE: between two large groups the number of
   cross-pairs explodes into the thousands, and OSNet's own already-
   documented weak discrimination on this dataset (same-camera-
   different-track similarity confirmed up to 0.96 between DIFFERENT
   real people) means a multiple-comparisons effect finds a spuriously
   high pair almost every time. Confirmed concretely: this collapsed 3
   visually-distinct real people (including a confirmed brown-fur-collar-
   coat person and a confirmed white-top person) into one 182-member
   group, even with a top-3-mean instead of raw max. Reverted to single-
   representative matching (the validated-good state); kept available
   behind `--use_group_pooled_similarity` (off by default) for a future,
   more discriminative ReID model.

**Result on POM camera_0000: 317 raw tracks -> 42 final groups** (down
from 119 with escalation alone, 193 with the one-shot boundary fix
alone), with ~93% precision in a 15-sample spot-check across the 3
largest resulting groups (14/15 consistent, 1 confirmed false merge --
a white-top person pulled into an otherwise dark-clothing-consistent
group). A large, real, carefully-verified improvement -- but still
roughly 4-6x the true ~7-10 people. Given four real bugs were found and
fixed (plus one found and correctly reverted) through this same rigorous
process, and the remaining gap traces to a DOCUMENTED, measured ReID-
model discrimination ceiling (not an unexamined assumption), the current
best understanding is that further improvement needs either a better/
domain-adapted ReID model or a fundamentally different signal (e.g. gait,
richer trajectory modeling) -- not more threshold or algorithm tuning on
top of OSNet's existing embeddings. Not yet rolled out to the full
pipeline (all 4 POM cameras, scene_061, EPFL-RLC) as of this writing --
see "Pending" note below.

**Pending as of this writing**: this fix (boundary veto + iterative
stitching + escalating threshold + pixel-based handoff signal, all ON by
default; group-pooled similarity OFF by default) has been validated on
POM camera_0000 alone. Rolling out to the full pipeline (all cameras in
all 3 scenes, with a full downstream rebuild -- graph, captions, actions
-- and a fresh scene_061 HOTA/AssA gate against the Phase 6a baseline of
27.02%/11.27%) is the natural next step but has not been done yet.

## Insights and lessons learned

- **Prompt-only fixes are unreliable; code-level/deterministic fixes are
  reliable.** Demonstrated concretely 6+ times: caption hallucination
  mitigation, single-candidate checking, fact corruption/caption bleeding,
  fabrication under compound queries, narration-without-action, and a
  "don't retype a long list" instruction that needed to be repeated INSIDE
  the tool's own report text, right next to the data, before it reliably
  worked -- a system-prompt-only version did not. The one consistent
  exception: nudges that are deterministic, code-generated, and placed
  immediately before the model's next turn (not just anywhere in the
  system prompt) DO work reliably -- recency/proximity in the context
  window matters more than where a rule is stated.
- **Greedy decoding is a double-edged choice.** Chosen for reproducibility
  (essential for this whole session's debugging methodology -- re-running
  an identical query after ONE code change and attributing any behavior
  difference to that change), but it's also more prone to
  repetition-induced degeneration than sampling, which is exactly what
  caused the tool-call-tag garbling under long, repetitive multi-call
  batches. The fix that held was capping batch size to the empirically
  reliable ceiling, not switching decoding strategy.
- **A 7B model's tool-calling has a real, measurable reliability ceiling**
  for this kind of agentic workload -- reliable at ~3 simultaneous tool
  calls per turn, degenerating and occasionally fabricating past ~6. A
  genuine capability limit, not a prompting problem; worth testing directly
  whether a larger or more heavily agentic-tool-use-trained model needs
  any of the batch-size/fabrication-scrubbing machinery built this session
  to work around it (see "Alternative models" below), rather than
  assuming either way.
- **Uncertainty should be surfaced as structured data, not just prose.**
  `caption_agreement`, `low_confidence` proximity flags,
  `MIN_RELIABLE_DETECTIONS`, `DUPLICATE_DETECTION_DISTANCE_M`, and now
  BLIP's own per-token confidence all follow the same pattern: expose
  uncertainty as its own field instead of hiding it inside a single point
  estimate, so downstream consumers (the LLM, or a human reading the
  report) can weight it appropriately.
- **Calibrate thresholds against the actual data distribution, not
  intuition.** Two guessed thresholds from an earlier session (BLIP
  confidence 0.5, and an implicit assumption that hard-excluding margin
  crops would be safe) both turned out to be substantially wrong when
  checked against real numbers (88% exclusion; 352/749 entities losing
  their caption entirely). Both were caught before shipping specifically
  because the project's convention is to verify empirically rather than
  trust a first-pass choice.
- **...but "empirical" only counts if it's the RIGHT population.** The
  crop-quality gate (captions/matching/actions work) repeated this exact
  mistake in a subtler form: it WAS calibrated against real numbers (1395
  real crops), but sampled via random (camera, frame) pairs -- a
  genuinely different, larger-skewing distribution than what
  `extract_entity_attributes.py` actually samples (crops spread across
  each entity's own trajectory). Result: 41% of entities wrongly flagged
  "unclear" despite the threshold being "empirically calibrated." Caught
  by comparing the real full-run output rate against the calibration's
  own predicted rate -- a discrepancy check worth doing even (especially)
  when a threshold was calibrated carefully, not just when it was
  guessed.
- **A tool's apparent finding is only as trustworthy as its naming and
  framing.** `list_non_human_entities` wasn't wrong in what it computed
  (absence of human words in a caption) -- it was wrong in what it CLAIMED
  that meant. Renaming it and rewriting every description honestly (once
  direct visual evidence contradicted the original premise) mattered as
  much as any code fix.
- **Aggregating a shared field for one purpose can silently break an
  unrelated downstream calculation that reads the same field.**
  `cross_camera_reid.py`'s within-camera consolidation step widened each
  group's frame range to min/max across all members -- reasonable-
  sounding on its own -- but that same field also feeds
  `plausibility_matrix`'s temporal-overlap check, and widening it
  spuriously triggered "overlap" against unrelated cross-camera
  candidates, collapsing spanning matches by ~90% before anyone noticed
  why. The bug wasn't in the aggregation logic itself (min/max IS a
  correct way to consolidate a frame range) -- it was in not tracing
  every OTHER place that field gets read before changing what it means.
  Caught by an A/B comparison (`--skip_same_camera` on identical inputs)
  that isolated the regression to the consolidation step specifically,
  not by reasoning about the aggregation in isolation.
- **A theoretically-sound extra signal can still net-hurt if it isn't
  discriminative enough for the actual population.** Phase 3's caption-
  similarity matching signal was well-motivated (per explicit user
  direction: match by appearance + wearing + position) and correctly
  implemented, but measurably increased over-merging on POM specifically
  -- not because the code was wrong, but because most people in this
  scene wear similarly generic dark clothing, so the signal's real-world
  entropy was too low to add discrimination, only false-positive bridges
  between different real people. The fix wasn't a bug fix -- it was
  running the actual before/after comparison (largest merged group's
  size and source diversity) and being willing to disable a feature that
  was implemented exactly as planned once the evidence didn't support it,
  rather than shipping it on the strength of the a-priori reasoning alone.
- **Searching more candidate pairs amplifies a noisy signal's false-
  positive tail, even when the aggregation looks more principled.**
  Phase 7's group-pooled-similarity fix (max, then top-3-mean, across all
  member pairs between two groups) was a correct diagnosis of a real bug
  (a single representative under-representing a large group) -- but
  checking thousands of cross-pairs between two large groups means even a
  well-calibrated model's rare high-similarity false positive (OSNet's
  own documented max 0.96 between confirmed different people) gets found
  reliably, not by chance. A signal that's trustworthy for ONE pairwise
  comparison isn't automatically trustworthy for "the best of N
  comparisons" -- N changes the effective false-positive rate even
  though no single comparison's threshold moved. Caught by direct visual
  inspection (a 182-member group provably containing 3 different real
  people), not by reasoning about the aggregation in the abstract --
  same pattern as the frame-range-widening bug from Phase 2.
- **RAG vs. this project's actual architecture**: only `search_by_appearance`
  is real RAG (embedding retrieval for a worded description -> candidate
  global_ids); everything else is deterministic graph traversal/aggregation
  over the NetworkX event graph. This matches the project's founding
  thesis (flat RAG alone can't handle compositional/temporal/cross-camera-
  identity queries) concretely, not just in principle.
- **MCP (Model Context Protocol) was discussed but not implemented.** The
  current tool-calling loop is a bespoke, in-process script (Qwen's native
  chat-template tool-call convention + a manual Python dispatch loop), not
  an MCP server/client. Wrapping `graph_tools.py`'s functions behind a real
  MCP server would make them reusable by any MCP-compatible agent, not
  just this one script's hard-wired 7B model -- a real, available option
  if broader tool reuse becomes a goal, and an interesting testbed for
  whether a more capable model actually needs the scaffolding built here
  specifically to work around Qwen2.5-7B's limits.

## Known limitations and future work

- **Scope is people-only by construction, not by accident.** The detector
  requests 5 COCO classes (`person, car, motorcycle, bus, truck`), and
  `build_pred_file.py` explicitly filters to `class == "person"` before
  the event graph is even built (this dataset's official eval is
  "Multi-Camera PEOPLE Tracking"). Static scene objects (e.g. warehouse
  boxes on a shelf, clearly visible in some frames) are never eligible to
  become a graph entity at any pipeline stage -- confirmed directly when
  asked why visible shelf boxes were never reported by any tool. A scope
  boundary, not a detection gap.
- **General movable-object tracking** (boxes, purses, backpacks, "anything
  that can move") is a stated future goal, not yet started. COCO already
  has `backpack`/`handbag`/`suitcase` classes the current YOLO checkpoint
  could detect cheaply by extending `CLASSES_OF_INTEREST` -- but "anything
  that can move" is inherently open-vocabulary, which no fixed class list
  covers; needs an open-vocabulary detector instead (see "Alternative
  models" below).
- **Basic action/behavior description is now implemented** (Phase 4,
  `extract_entity_actions.py` -- `action_description`/`action_agreement`
  fields, `search_by_action`/`get_entity_action` tools), but scoped to
  coarse locomotion/posture (walking, standing still, sitting, carrying)
  rather than specific EVENTS (a man turned on a light, pressed a button,
  entered an elevator) -- those still aren't representable, and this
  dataset's own scenes (an outdoor terrace, an indoor hallway) don't
  contain literal instances of the original illustrative examples anyway.
  Real, unfixed hallucination risk confirmed by direct video inspection:
  one entity was confidently reported "sitting" (agreement=1.0) when the
  same clip's real frames clearly showed the person walking -- action
  description is measurably LESS reliable than appearance captioning
  (17.3% low-agreement vs. 0.85%), and unlike Phase 1's color handling,
  no deterministic scrub exists yet for either this class of hallucination
  or the confirmed appearance-word leakage into action text (general
  clothing vocabulary is too large/fuzzy for a `strip_color_words`-style
  fix). Treat `action_description` as the least-trustworthy field in this
  graph schema until this is addressed further.
- **Within-camera track fragmentation is still the dominant identity-count
  error**, confirmed quantitatively: 20 true identities vs. 809 predicted,
  with within-camera fragmentation (not cross-camera matching) as the
  measured dominant cause. Properly fixing this needs appearance-based
  association during single-camera tracking (e.g. BoT-SORT+ReID), not a
  parameter tweak (a `track_buffer` increase was tried and didn't help,
  confirming the cause is per-frame ID-swap ambiguity, not track loss over
  time) -- deferred, not fixed.
- **OSNet's domain gap on this synthetic dataset is real and unaddressed**:
  same-camera-different-track cosine similarity (a clean "different real
  person" baseline) averaged ~0.70 with max up to 0.96 -- weak
  discrimination, since the checkpoint was fine-tuned on real photos
  (Market-1501), not synthetic renders. Would need domain adaptation or a
  different backbone to close.
- **Grounding/consistency gaps in the LLM's own free text remain possible**
  even after all the deterministic-report machinery -- e.g. the model's
  short framing sentence has been observed to misattribute a fact to the
  wrong entity, while the guaranteed-correct report right below it stays
  accurate. The deterministic-report pattern makes this cosmetic rather
  than fact-threatening, but hasn't eliminated it at the source.
- **The captioning pipeline's remaining failure mode**: coherent-but-wrong
  captions where NONE of the 3 sampled crops mention a person -- majority
  vote can't fix a unanimous wrong answer. No current signal catches this
  class; would need more crops per entity, a differently-hallucination-
  prone captioner, or cross-referencing the detector's own class
  confidence.

## Alternative models to consider

Not implemented -- options identified for when/if revisiting each stage.

- **Detection, for open-vocabulary/movable-object tracking**: SAM3
  (promptable segmentation with text/visual prompts, could name arbitrary
  objects like "box" or "purse" that aren't COCO classes), Grounding DINO
  or OWL-ViT (open-vocabulary detection via text prompts), or YOLO-World
  (open-vocabulary YOLO variant, closer to a drop-in replacement for the
  current `ultralytics` detector). These trade detection speed for
  vocabulary flexibility -- worth benchmarking against this project's
  real-time-per-camera budget before committing.
- **Captioning, as a BLIP replacement**: BLIP-2 or InstructBLIP (stronger
  language grounding, still efficient), Florence-2 (Microsoft, strong at
  short factual captions, small), or a small Qwen-VL/Qwen2.5-VL variant
  (multimodal sibling of the LLM already used here, so the project would
  only depend on one model family for both captioning and query reasoning
  -- worth checking whether its hallucination profile is actually better
  than BLIP's documented "sandwich"/repetition failures before switching,
  not assuming it).
- **Video-level captioning/action recognition**: Qwen2-VL / Qwen2.5-VL can
  take multiple frames or short clips directly (unlike BLIP, one static
  image at a time) -- a natural first thing to try for "did this entity
  perform action X" queries, since it wouldn't require a wholly new model
  family. VideoMAE or SlowFast are purpose-built temporal-action models if
  a dedicated (non-VLM) action-recognition stage is preferred instead.
- **Query-interface LLM**: a larger Qwen2.5 variant (14B/32B) or a model
  more heavily trained for agentic tool-use specifically. Directly
  motivated by this session's finding that Qwen2.5-7B has a measured
  ~3-call reliable batch ceiling under greedy decoding -- worth testing
  whether a bigger/more tool-hardened model removes the need for the
  batch-cap/fabrication-scrubber scaffolding built to work around it,
  rather than assuming it would.
- **ReID, for the confirmed domain gap**: a checkpoint fine-tuned on
  synthetic/rendered pedestrian data (if one exists) instead of
  Market-1501 (real photos), or domain adaptation on top of the current
  OSNet checkpoint using this dataset's own synthetic crops.

## Tests

`tests/` (stdlib `unittest`, no new dependency -- consistent with keeping
`mct-env` slim). Three files, split by cost/dependency:

- `tests/test_regressions.py` -- fast (<1s), pure Python logic, no GPU/
  model/graph file needed. One test class per bug found across sessions
  (degenerate captions, human-word heuristic gaps, agreement-score
  display, fabrication scrubbing, report-merging, display-report
  assembly, invented-id recovery, the interactor-count nudge, narration-
  without-action, tool-call parsing, caption-confidence calibration --
  renamed from BLIP- to CAPTION_CONFIDENCE_THRESHOLD when the model
  changed). Extended for the captions/matching/actions work: hyphenated
  color-word stripping, dominant-color naming sanity (including a BGR/
  RGB channel-order check), the crop-quality area-threshold recalibration
  (with an explicit test that a narrow-but-tall crop ISN'T wrongly
  rejected -- the actual bug), the same-camera-vs-cross-camera
  plausibility veto asymmetry (the single highest-risk detail from
  planning), the frame-range-widening regression (encodes the exact
  mechanism that collapsed cross-camera matches from 121 to 13), and the
  caption-similarity additive-combination math. Further extended for
  Phase 7's fragmentation deep-dive: the boundary-touch overlap grace
  window, the clean-handoff signal overriding noisy appearance similarity
  (including the linear-decay-from-zero and post-plausibility-multiplication
  sub-bugs), iterative multi-round chain resolution (a synthetic 4-track
  chain that a single Hungarian round can't fully resolve), and a same-
  camera diagonal-exclusion bug caught BY writing these tests (fired on
  `same_camera=True` alone rather than checking `keys_a is keys_b`,
  harmless in production but wrong in general). Run this one on every
  change.
- `tests/test_graph_integration.py` -- loads the real
  `event_graph_with_attrs.gpickle` (seconds to ~3 minutes -- one test does
  a real full-graph scan). Verifies the fixes hold against actual data:
  entity 8 is no longer captioned "sandwich", `find_nearby_entities`
  isn't silently truncated at 10, no flagged entity has a degenerate
  caption or a too-short track, proximity match-type categorization is
  internally consistent, ranking isn't capped at 10. Also covers the new
  action-description tools: graceful (no crash, empty/error result, not
  an exception) behavior on scene_061's graph (hasn't run Phase 4 yet as
  of this writing) via `TestActionToolsGracefulOnGraphWithoutActions`,
  plus real end-to-end checks against POM's graph (which has run Phase 4)
  via `PomActionToolsTestCase`.
- `tests/test_llm_smoke.py` -- SLOW, GPU-dependent, skipped unless
  `RUN_LLM_TESTS=1` is set (loads Qwen2.5-7B-Instruct and runs real
  generation; each query can take minutes). Covers the two bugs that are
  fundamentally about model behavior, not deterministic code: the
  batch-size/fabrication saga, and invented-id recovery in a live run.

Run everything fast: `python tests/test_regressions.py -v`
Run graph integration: `python tests/test_graph_integration.py -v`
Run LLM smoke tests: `RUN_LLM_TESTS=1 python tests/test_llm_smoke.py -v`

## Final pipeline summary (scene_061, cameras 0535-0538)

Detection+tracking (`yolo11s`@conf=0.15) -> world-coordinate projection
(homography inverse) -> identity resolution (OSNet-Market1501 appearance
+ within-camera stitching, same_camera_threshold=0.75, THEN cross-camera
Hungarian per-camera-pair matching, threshold=0.8) -> event graph
(NetworkX) -> Qwen2.5-VL-3B constrained-prompt appearance captions (RGB/
Lab color sampling spliced in, never the VLM's own guess) + action
descriptions (native video input) -> Qwen2.5-7B-Instruct tool-calling
query layer (search_by_appearance + search_by_action, among others).

Headline numbers: single-camera HOTA 24-41% (varies by crowd density);
full 4-camera HOTA 21.30% with no merging -> 26.90% with cross-camera-
only ReID -> **27.02% (AssA 11.27%) with within-camera stitching added**
(captions/matching/actions work) -- 20 true identities vs. 809 (cross-
camera-only baseline) vs. 364 (with within-camera stitching), still
dominated by within-camera fragmentation but a real, ground-truth-
verified improvement, not just a lower raw count; 4/6 curated
compositional queries answered correctly and conservatively (pre-dates
the captions/matching/actions work -- due for a fresh eval-set pass on
the current pipeline).

Environment note: `mct-env` venv is set up with `huggingface_hub`,
`opencv-python-headless`, `networkx`, `ultralytics` (which pulls in a
CUDA-enabled `torch` matching the RTX 5090 automatically) plus
`pandas`/`scipy`/`matplotlib` for the vendored TrackEval code. Added for
the captions/matching/actions work: `qwen-vl-utils` + `av` (Qwen2.5-VL's
video input) and `einops` (a Molmo dependency -- kept even though Molmo
itself was ultimately not used, since it's small and harmless). Molmo's
full loading also wanted `tensorflow` (for a static import-check in its
`trust_remote_code` file, not actually used at runtime) -- deliberately
NOT installed, to keep this env slim; see Phase 0 above.
