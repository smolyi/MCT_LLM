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
