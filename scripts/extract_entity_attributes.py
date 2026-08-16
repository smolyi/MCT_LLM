"""
Attribute extraction: for each entity in the event graph, sample several
crops spread across ALL its sightings (not just one), caption each, and
store both a consensus caption and an agreement score as node attributes.

Single-crop captioning (see git history) was a documented source of real
errors this project found directly by inspecting actual video frames: e.g.
one entity's sole crop got captioned "a woman in a red dress is playing
tennis" when the crop actually showed a person in a red TOP + jeans
standing on a plain floor -- there is no tennis anywhere in this dataset,
and BLIP free-associated it from a blue grid-tiled floor texture. A single
bad/ambiguous/partial crop had no way to be caught or corrected before this.

Multiple crops let disagreement itself become a signal, following the same
"uncertainty is information" pattern already used for tracking (low_confidence
proximity matches, spatio-temporal ReID plausibility): if several crops of
the same entity produce very different captions, that's useful information
(this entity's description is unreliable), not noise to average away.

Consensus caption = the sampled caption whose sentence-embedding is closest
to the mean of all sampled captions' embeddings (the "medoid") -- more
robust than an arbitrary single crop, and avoids trying to average
embeddings and decode the result, which the captioner can't do (not
invertible). caption_agreement = mean pairwise cosine similarity across the
sampled captions' embeddings, in [0, 1] -- low agreement flags a caption as
contested even though the query interface can't independently verify it.

Model: Qwen2.5-VL-3B-Instruct, chosen via a real Phase 0 bake-off against
LLaVA-OneVision, BLIP-2/InstructBLIP, and (blocked by a tensorflow
dependency in its trust_remote_code file) Molmo -- see CLAUDE.md's
"Additional data sources" / captions-matching-actions section for the
full writeup. Superseded the original Salesforce/blip-image-captioning-base
(../LMM_dive's TinyLMM was ruled out earlier for its own reasons -- 0%
"no"-accuracy on POPE) specifically because BLIP-base has NO instruction
channel at all (its `text=` argument is a caption PREFIX to continue
writing from, not a directive it can obey) -- that structural gap, not
just "BLIP is weak," was the actual mechanism behind the tennis/sandwich
hallucinations: there was no way to tell it "describe appearance only."

Two deterministic (not prompt-reliant) fixes layered on top of the
constrained instruction prompt, both added after the bake-off showed
prompting alone isn't reliable enough on its own:
- crop_quality.py's blur/size gate skips the VLM call entirely for crops
  too degenerate to read (the bake-off's one real failure case was an 8px-
  wide sliver, almost certainly a false-positive detection, that 2 of 3
  candidates confidently captioned anyway despite an explicit "say if
  unclear" hedge).
- color_utils.py's RGB/Lab dominant-color sampling replaces the VLM's own
  color guessing entirely (also unreliable even under an explicit "do not
  mention color" instruction, confirmed in the same bake-off) -- garment
  color in the final appearance_caption always comes from actual pixels,
  never the captioning model.

Usage:
  python scripts/extract_entity_attributes.py --scene_dir data/scene_061 --graph data/scene_061/event_graph.gpickle \
      --out data/scene_061/event_graph_with_attrs.gpickle
"""
import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

from color_utils import garment_region, dominant_color_name, strip_color_words
from crop_quality import is_low_quality_crop, UNCLEAR_CAPTION
from graph_tools import GraphTools
from video_source import resolve_video_source

CAPTION_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"

CONSTRAINED_PROMPT = (
    "Describe ONLY the visible clothing type and accessories of the person in this image "
    "(e.g. jacket, t-shirt, backpack, hat). Do NOT mention color. Do NOT describe any action, "
    "activity, pose, or what the person might be doing. If something is unclear or not visible, "
    "say so briefly instead of guessing."
)  # validated in the Phase 0 bake-off: eliminated activity/scene hallucination in 18/18 real crops
# across all 3 candidates tested. Color suppression is NOT reliable even with this instruction (also
# confirmed in the bake-off) -- handled by strip_color_words()+color_utils instead of trusted here.

CROPS_PER_ENTITY = 3

FRAME_MARGIN_PX = 5  # a bbox this close to the frame edge is likely a cut-off figure -- captioning a
# partial view (only half a person visible, say) is far more likely to produce a wrong/nonsense
# caption than a fully-visible crop would, so these are excluded from crop sampling entirely.

MIN_SIGHTING_DETECTIONS_FOR_CAPTION = 10  # skip sampling crops from a sighting fragment this short --
# same threshold GraphTools.MIN_RELIABLE_DETECTIONS uses for proximity matching, same reasoning: a
# handful of detections is more likely a track fragment/artifact than a real, stable sighting worth
# captioning at all.

CAPTION_CONFIDENCE_THRESHOLD = 0.3  # mean per-token max-softmax probability across the generated
# caption. Carried over from BLIP's original 0.3 (~p10 of BLIP's own 0.316-0.879 confidence spread)
# as a starting guess, then CHECKED (not assumed) against Qwen2.5-VL-3B's real distribution on 193
# real POM captions: min=0.921, p10=0.975, mean=0.992 -- the signal doesn't transfer, not just the
# threshold. Qwen2.5-VL-3B's greedy decoding on this short, templated, low-entropy task ("The person
# is wearing a [garment]...") is almost always sharply peaked, unlike BLIP's more variable open-ended
# captioning -- there is no meaningful low-confidence tail to gate on here. Picking an arbitrary cutoff
# inside the observed 0.92-1.0 band would exclude "least confident of many good outputs" rather than
# genuine failures, so 0.3 is left in place deliberately AS AN EFFECTIVELY-INERT SAFETY NET (never
# expected to fire in practice for this model) rather than replaced with a fake-precise number. The
# degenerate-caption check and cross-crop caption_agreement remain the two signals that actually do
# work for this model.


def is_degenerate_caption(caption: str) -> bool:
    """Flags BLIP decoder degeneration -- greedy decoding occasionally gets stuck looping the same
    word (e.g. "a bald bald bald bald..."), a real observed failure -- found via manual inspection of
    an entity whose consensus caption was exactly this, which the medoid step had picked over two
    coherent, real person descriptions in its own all_captions ("a woman walking down the street with
    her dog", "a man in a white shirt and tie") simply because its embedding happened to land near
    their mean. Same root-cause class as this project's LLM tool-call generation degenerating under
    greedy decoding (see query_interface.py) -- repetitive output collapsing decoding reliability,
    just in a captioning model instead of an LLM. A caption is degenerate if any single word repeats
    3+ times in a row."""
    words = caption.lower().split()
    run = 1
    for i in range(1, len(words)):
        if words[i] == words[i - 1]:
            run += 1
            if run >= 3:
                return True
        else:
            run = 1
    return False


def pick_sample_frames(frames: list, k: int) -> list:
    """k evenly-spaced (camera, frame) pairs from a list sorted by frame number, dedup'd."""
    if len(frames) <= k:
        return frames
    idxs = np.linspace(0, len(frames) - 1, k).round().astype(int)
    seen, out = set(), []
    for i in idxs:
        if frames[i] not in seen:
            seen.add(frames[i])
            out.append(frames[i])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_dir", type=str, required=True)
    parser.add_argument("--graph", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--crops_per_entity", type=int, default=CROPS_PER_ENTITY)
    parser.add_argument("--max_entities", type=int, default=None,
                         help="Cap the number of entities processed, for a fast smoke test before a full run.")
    args = parser.parse_args()

    scene_dir = Path(args.scene_dir)
    with open(args.graph, "rb") as f:
        G = pickle.load(f)

    # Gather every sighting per entity, across ALL cameras -- not just the single largest one.
    sightings_by_entity = defaultdict(list)  # gid -> [(node_id, data), ...]
    for n, d in G.nodes(data=True):
        if d["type"] == "sighting":
            sightings_by_entity[d["global_id"]].append((n, d))

    if args.max_entities is not None:
        keep_gids = set(list(sightings_by_entity.keys())[:args.max_entities])
        sightings_by_entity = {gid: v for gid, v in sightings_by_entity.items() if gid in keep_gids}

    print(f"Sampling up to {args.crops_per_entity} crops each for {len(sightings_by_entity)} entities...")

    # Read pred_full.txt once, index by (camera, frame) -> list of (gid, l, t, w, h)
    pred_path = scene_dir / "pred_full.txt"
    by_cam_frame = defaultdict(list)
    with open(pred_path) as f:
        for line in f:
            cam, gid, frame, l, t, w, h, wx, wy = line.split()
            by_cam_frame[(cam, int(frame))].append((int(gid), float(l), float(t), float(w), float(h)))

    # For each entity, pick sample (camera, frame) points spread across ALL its sightings -- but only
    # from sightings with enough detections to be a real, stable sighting rather than a track fragment
    # (see MIN_SIGHTING_DETECTIONS_FOR_CAPTION). If an entity has no sighting that qualifies, fall back
    # to using all of them anyway -- some sample is still better than captioning nothing.
    sample_targets_by_camera = defaultdict(list)  # camera -> [(gid, frame), ...]
    for gid, sightings in sightings_by_entity.items():
        reliable = [(n, d) for n, d in sightings if d.get("num_detections", 0) >= MIN_SIGHTING_DETECTIONS_FOR_CAPTION]
        source_sightings = reliable or sightings
        all_frames = []  # (frame, camera) -- sort key first so spacing is chronological-ish
        for n, d in source_sightings:
            for frame, t, wx, wy in d.get("trajectory", []):
                all_frames.append((frame, d["camera"]))
        all_frames.sort(key=lambda x: x[0])
        for frame, cam in pick_sample_frames(all_frames, args.crops_per_entity):
            sample_targets_by_camera[cam].append((gid, frame))

    # Extract crops, grouped by camera so each video is opened/read once. Split into two phases:
    # first resolve every target's exact bbox (pure dict lookups against by_cam_frame, no video I/O),
    # THEN read the video in ONE SEQUENTIAL pass collecting every needed frame as it's encountered --
    # never cap.set()-seeking per target. This matters a lot more than it sounds: measured directly on
    # an old (2008, IV50-codec) AVI source, a single cap.set() random seek took ~0.83s, vs ~0.0016s for
    # a sequential read -- ~500x slower, which at ~3 crops x hundreds of entities turned a
    # multi-second job into a multi-hour one. Modern H.264/MP4 sources (e.g. scene_061) seek fast
    # enough that this wasn't previously noticeable, but sequential reading is strictly no worse for
    # them either, so this applies uniformly rather than being source-specific.
    crops_by_gid = defaultdict(list)  # gid -> [np.ndarray, ...]
    for cam, targets in sample_targets_by_camera.items():
        cam_dir = f"camera_{int(cam):04d}"
        cap = cv2.VideoCapture(resolve_video_source(scene_dir / cam_dir))
        frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Phase 1: resolve each (gid, frame) target to an exact (frame, bbox) -- no video I/O yet.
        resolved = {}  # gid -> (fr, l, t, w, h)
        for gid, frame in targets:
            # Two passes: prefer a non-margin bbox within the search window, but fall back to a
            # margin-touching one rather than yielding nothing at all -- a first version of this made
            # margin-avoidance a hard requirement within the ±16-frame window, which left 352 of 749
            # entities (mostly ones that only ever appear near a frame edge, e.g. walking along the
            # periphery) with ZERO extractable crops and therefore no appearance_caption at all, a much
            # worse outcome than occasionally captioning a partially cut-off figure.
            bbox = None
            fallback_bbox = None
            for offset in range(0, 16):
                for fr in (frame + offset, frame - offset):
                    for cand_gid, l, t, w, h in by_cam_frame.get((cam, fr), []):
                        if cand_gid != gid:
                            continue
                        touches_margin = (l <= FRAME_MARGIN_PX or t <= FRAME_MARGIN_PX
                                           or (l + w) >= frame_w - FRAME_MARGIN_PX
                                           or (t + h) >= frame_h - FRAME_MARGIN_PX)
                        if touches_margin:
                            if fallback_bbox is None:
                                fallback_bbox = (fr, l, t, w, h)
                            continue  # keep searching nearby frames for a clean, non-margin option
                        bbox = (fr, l, t, w, h)
                        break
                    if bbox:
                        break
                if bbox:
                    break
            bbox = bbox or fallback_bbox
            if bbox is not None:
                resolved[gid] = bbox

        # Phase 2: one sequential pass over the video, grabbing every resolved frame as it comes up.
        targets_by_frame = defaultdict(list)  # frame -> [(gid, l, t, w, h), ...]
        for gid, (fr, l, t, w, h) in resolved.items():
            targets_by_frame[fr].append((gid, l, t, w, h))
        if targets_by_frame:
            last_needed_frame = max(targets_by_frame)
            frame_idx = 0
            while frame_idx <= last_needed_frame:
                ok, img = cap.read()
                if not ok:
                    break
                if frame_idx in targets_by_frame:
                    h_img, w_img = img.shape[:2]
                    for gid, l, t, w, h in targets_by_frame[frame_idx]:
                        x1, y1 = max(0, int(l)), max(0, int(t))
                        x2, y2 = min(w_img, int(l + w)), min(h_img, int(t + h))
                        if x2 > x1 and y2 > y1:
                            crops_by_gid[gid].append(img[y1:y2, x1:x2])  # BGR, as read by cv2 -- kept
                            # native for color_utils/crop_quality (both cv2-based); converted to RGB
                            # only at the point of PIL image construction for the VLM, below.
                frame_idx += 1
        cap.release()

    n_crops = sum(len(v) for v in crops_by_gid.values())
    print(f"Extracted {n_crops} crops across {len(crops_by_gid)} entities, running {CAPTION_MODEL_ID} captioning...")

    processor = AutoProcessor.from_pretrained(CAPTION_MODEL_ID)
    vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(CAPTION_MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda")
    vlm.eval()
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    human_words = GraphTools.HUMAN_CAPTION_WORDS

    def caption_one(crop_bgr: np.ndarray) -> tuple:
        """Returns (caption_text, confidence). Deterministic crop-quality gate runs FIRST -- a crop
        that fails it never reaches the VLM at all, per the Phase 0 bake-off finding that models can't
        be trusted to self-report uncertainty via prompting alone."""
        if is_low_quality_crop(crop_bgr):
            return UNCLEAR_CAPTION, 1.0  # confidence 1.0: this is a deterministic code decision, not
            # a generation the confidence-gate below should second-guess.
        upper = dominant_color_name(garment_region(crop_bgr, "upper"))
        lower = dominant_color_name(garment_region(crop_bgr, "lower"))
        image = Image.fromarray(crop_bgr[:, :, ::-1])  # BGR -> RGB for the VLM
        messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                                   {"type": "text", "text": CONSTRAINED_PROMPT}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = vlm.generate(**inputs, max_new_tokens=60, return_dict_in_generate=True, output_scores=True)
        gen = out.sequences[:, inputs["input_ids"].shape[1]:]
        garment_text = processor.batch_decode(gen, skip_special_tokens=True)[0].strip()
        garment_text = strip_color_words(garment_text)  # the VLM's "no color" instruction is NOT
        # reliable (confirmed in the bake-off) -- stripped here so it can never contradict the
        # RGB-sampled colors below, rather than trusting the prompt alone.
        token_probs = [torch.softmax(step_logits[0], dim=-1).max().item() for step_logits in out.scores]
        confidence = sum(token_probs) / len(token_probs) if token_probs else 0.0
        caption = f"{upper} top, {lower} bottom. {garment_text}"
        return caption, confidence

    n_captioned = 0
    n_low_agreement = 0
    for gid, crops in crops_by_gid.items():
        captions = []
        caption_confidences = []
        for crop in crops:
            caption, confidence = caption_one(crop)
            captions.append(caption)
            caption_confidences.append(confidence)

        # A caption is excluded from medoid selection (never eligible as the consensus) if it's
        # degenerate (decoder looping) OR the model itself generated it with low confidence -- still
        # recorded in all_captions/all_caption_confidences either way, just not trusted as consensus.
        valid_idxs = [i for i, c in enumerate(captions)
                      if not is_degenerate_caption(c) and caption_confidences[i] >= CAPTION_CONFIDENCE_THRESHOLD]
        if not valid_idxs:
            valid_idxs = [0]  # every sample failed both checks -- no usable signal, but still need
            # something to store as appearance_caption
        valid = [captions[i] for i in valid_idxs]
        some_excluded = len(valid) < len(captions)

        # Majority-vote tiebreak: found via manual inspection of a real case (global_id 8, "a sandwich
        # with a sandwich on it") where pure embedding-distance-to-mean picked a caption with no human
        # words as consensus even though 2 of 3 valid captions clearly described a person ("a man is
        # standing on a ledge", "a woman is running on a black surface") -- a small-sample (n=3) medoid
        # can get pulled off by one confident-but-wrong outlier. If a strict majority of the valid
        # captions mention a person, restrict medoid candidates to just those before picking.
        def _mentions_human(c):
            return bool(set(c.lower().replace(",", " ").replace(".", " ").split()) & human_words)
        human_subset = [c for c in valid if _mentions_human(c)]
        if len(human_subset) > len(valid) - len(human_subset):
            valid = human_subset

        if len(valid) == 1:
            # Either only one crop was sampled (nothing to disagree with, full confidence) or every
            # OTHER sample was excluded (real uncertainty, even though one caption survives to compare).
            consensus, agreement = valid[0], (0.0 if some_excluded else 1.0)
        else:
            embs = embedder.encode(valid, normalize_embeddings=True)
            mean_emb = embs.mean(axis=0)
            mean_emb /= np.linalg.norm(mean_emb)
            consensus = valid[int(np.argmax(embs @ mean_emb))]
            n = len(valid)
            pairwise_sum = (embs @ embs.T).sum() - n  # exclude self-similarity (diagonal = 1)
            agreement = float(pairwise_sum / (n * (n - 1)))

        G.nodes[f"entity:{gid}"]["appearance_caption"] = consensus
        G.nodes[f"entity:{gid}"]["all_captions"] = captions
        G.nodes[f"entity:{gid}"]["all_caption_confidences"] = [round(c, 3) for c in caption_confidences]
        G.nodes[f"entity:{gid}"]["caption_agreement"] = round(agreement, 3)
        n_captioned += 1
        if agreement < 0.6:
            n_low_agreement += 1
        if n_captioned % 100 == 0:
            print(f"  ...{n_captioned}/{len(crops_by_gid)} ({n_low_agreement} low-agreement so far)")

    print(f"Captioned {n_captioned} entities ({n_low_agreement} with caption_agreement < 0.6)")

    # Empirical check of CAPTION_CONFIDENCE_THRESHOLD's carryover-from-BLIP value, same discipline as
    # the original BLIP 0.5->0.3 recalibration: print the REAL distribution so the constant above can
    # be corrected if Qwen2.5-VL-3B's confidence calibration turns out to differ substantially.
    all_confs = [c for gid in crops_by_gid for c in G.nodes[f"entity:{gid}"]["all_caption_confidences"]]
    if all_confs:
        arr = np.array(all_confs)
        pct_excluded = float((arr < CAPTION_CONFIDENCE_THRESHOLD).mean() * 100)
        print(f"Caption confidence distribution: mean={arr.mean():.3f} median={np.median(arr):.3f} "
              f"p10={np.percentile(arr, 10):.3f} max={arr.max():.3f} -- "
              f"{pct_excluded:.1f}% of all captions fall below CAPTION_CONFIDENCE_THRESHOLD={CAPTION_CONFIDENCE_THRESHOLD}")

    with open(args.out, "wb") as f:
        pickle.dump(G, f)
    print(f"Saved graph with attributes to {args.out}")


if __name__ == "__main__":
    main()
