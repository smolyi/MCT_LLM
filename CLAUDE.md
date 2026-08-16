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
  intuition.** Two guessed thresholds this session (BLIP confidence 0.5,
  and an implicit assumption that hard-excluding margin crops would be
  safe) both turned out to be substantially wrong when checked against
  real numbers (88% exclusion; 352/749 entities losing their caption
  entirely). Both were caught before shipping specifically because the
  project's convention is to verify empirically rather than trust a
  first-pass choice.
- **A tool's apparent finding is only as trustworthy as its naming and
  framing.** `list_non_human_entities` wasn't wrong in what it computed
  (absence of human words in a caption) -- it was wrong in what it CLAIMED
  that meant. Renaming it and rewriting every description honestly (once
  direct visual evidence contradicted the original premise) mattered as
  much as any code fix.
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
- **Specific action/event recognition** (a man turned on a light, pressed a
  button, entered an elevator) is a stated future goal and a genuinely
  different KIND of problem from everything built so far. BLIP captions
  one static crop; YOLO detects one static frame; the event graph tracks
  POSITION over time. None of that can see a state change or a brief
  interaction. Needs either a real temporal/video-action-recognition
  model, or a VLM prompted over short clips/frame sequences -- a new
  pipeline stage, not an extension of BLIP captioning.
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
  model/graph file needed. One test class per bug found this session
  (degenerate captions, human-word heuristic gaps, agreement-score
  display, fabrication scrubbing, report-merging, display-report
  assembly, invented-id recovery, the interactor-count nudge, narration-
  without-action, tool-call parsing, BLIP-confidence calibration). Run
  this one on every change.
- `tests/test_graph_integration.py` -- loads the real
  `event_graph_with_attrs.gpickle` (seconds to ~3 minutes -- one test does
  a real full-graph scan). Verifies the fixes hold against actual data:
  entity 8 is no longer captioned "sandwich", `find_nearby_entities`
  isn't silently truncated at 10, no flagged entity has a degenerate
  caption or a too-short track, proximity match-type categorization is
  internally consistent, ranking isn't capped at 10.
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
(homography inverse) -> cross-camera ReID (OSNet-Market1501 + Hungarian
per-camera-pair matching, threshold=0.8) -> event graph (NetworkX) ->
BLIP attribute captions -> Qwen2.5-7B-Instruct tool-calling query layer.

Headline numbers: single-camera HOTA 24-41% (varies by crowd density);
full 4-camera HOTA 21.30% with no cross-camera merging -> 26.90% with
ReID-based merging; 20 true identities vs. 809 predicted (dominated by
within-camera fragmentation, not cross-camera error); 4/6 curated
compositional queries answered correctly and conservatively.

Environment note: `mct-env` venv is set up with `huggingface_hub`,
`opencv-python-headless`, `networkx`, `ultralytics` (which pulls in a
CUDA-enabled `torch` matching the RTX 5090 automatically) plus
`pandas`/`scipy`/`matplotlib` for the vendored TrackEval code.
