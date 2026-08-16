"""
Action description: for each entity in the event graph, sample one or two
short CLIPS (consecutive frames, not static crops) from its reliable
sightings, describe the physical behavior visible across each clip with
Qwen2.5-VL-3B's native video input, and store a consensus action
description + agreement score as node attributes -- structurally mirroring
extract_entity_attributes.py's appearance_caption/caption_agreement
pattern, but for BEHAVIOR instead of appearance.

Kept as its own script rather than folded into extract_entity_attributes.py:
action-prompt iteration will need several fast re-runs while the slower
appearance stage shouldn't need to re-run every time; merging them is a
reasonable later cleanup once the prompt has stabilized.

Clip sampling, not static crops: a single frame can't show a behavior, only
a pose. 8-16 consecutive frames (~0.3-0.5s at 30fps) per reliable sighting,
verified directly against real footage (a 12-frame walking clip correctly
produced "Walking", not an invented interaction). Still two-phase, same
lesson as every other crop-extraction step in this project (~500x random-
seek-vs-sequential-read gap): resolve every clip's exact frame RANGE and
per-frame bbox first (pure dict lookups against pred_full.txt -- bbox moves
slightly frame to frame within a clip, so it's looked up per-frame, not
reused as one static box), THEN one sequential pass per camera collecting
every needed frame.

Reuses Phase 1's crop-quality gate (crop_quality.is_low_quality_crop) per
frame -- a clip majority-built from unreadable slivers shouldn't be trusted
any more than a single bad static crop would be. Constrained action prompt
explicitly forbids appearance claims (the inverse of Phase 1's appearance
prompt forbidding activity claims) -- verified in the same spirit as the
Phase 0 bake-off: don't invent an interaction when nothing notable is
happening, say "walking" or "standing still" instead.

Usage:
  python scripts/extract_entity_actions.py --scene_dir data/POM/terrace1 --graph data/POM/terrace1/event_graph_with_attrs.gpickle \
      --out data/POM/terrace1/event_graph_with_attrs.gpickle
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
from qwen_vl_utils import process_vision_info

from crop_quality import is_low_quality_crop
from extract_entity_attributes import is_degenerate_caption, MIN_SIGHTING_DETECTIONS_FOR_CAPTION
from video_source import resolve_video_source

CAPTION_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
CLIPS_PER_ENTITY = 2
CLIP_LENGTH_FRAMES = 12  # within the 8-16 (~0.3-0.5s @ 30fps) range verified against real footage
MIN_GOOD_FRAMES_FRACTION = 0.5  # a clip needs at least half its frames to pass the crop-quality
# gate to be trusted at all -- otherwise skipped, same "don't guess on unreadable input" principle
# as Phase 1, applied per-clip instead of per-crop.

ACTION_CONSTRAINED_PROMPT = (
    "Describe ONLY the physical action/behavior of the person across these frames (e.g. walking, "
    "running, standing still, sitting, carrying something, interacting with another person). Do NOT "
    "describe clothing, appearance, or color. If nothing notable is happening, say \"walking\" or "
    "\"standing still\" rather than inventing an interaction."
)  # verified directly: a real 12-frame walking clip produced "Walking", not an invented interaction.

UNCLEAR_ACTION = "action unclear (low-quality clip)"

ACTION_CONFIDENCE_THRESHOLD = 0.3  # same starting value and same re-derivation caveat as
# extract_entity_attributes.py's CAPTION_CONFIDENCE_THRESHOLD -- Qwen2.5-VL-3B's confidence
# calibration for THIS task (video input, action vocabulary) has not been separately checked
# against a real distribution yet; re-derive empirically before trusting this as a real gate,
# same discipline as every other threshold in this project.


def pick_clip_starts(sightings: list, clip_length: int, k: int) -> list:
    """Up to k (camera, start_frame, end_frame) clip windows spread across an entity's reliable
    sightings -- one clip per sighting, centered on that sighting's own trajectory, capped at k
    sightings (largest first, a proxy for most reliable) if more are available."""
    sightings_sorted = sorted(sightings, key=lambda nd: -nd[1].get("num_detections", 0))
    windows = []
    for n, d in sightings_sorted[:k]:
        traj = d.get("trajectory", [])
        if not traj:
            continue
        frames = [t[0] for t in traj]
        mid_idx = len(frames) // 2
        start = frames[mid_idx]
        end_idx = min(mid_idx + clip_length, len(frames)) - 1
        end = frames[end_idx]
        windows.append((d["camera"], start, end))
    return windows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_dir", type=str, required=True)
    parser.add_argument("--graph", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--clips_per_entity", type=int, default=CLIPS_PER_ENTITY)
    parser.add_argument("--clip_length", type=int, default=CLIP_LENGTH_FRAMES)
    parser.add_argument("--max_entities", type=int, default=None,
                         help="Cap the number of entities processed, for a fast smoke test before a full run.")
    args = parser.parse_args()

    scene_dir = Path(args.scene_dir)
    with open(args.graph, "rb") as f:
        G = pickle.load(f)

    sightings_by_entity = defaultdict(list)
    for n, d in G.nodes(data=True):
        if d["type"] == "sighting":
            sightings_by_entity[d["global_id"]].append((n, d))

    if args.max_entities is not None:
        keep_gids = set(list(sightings_by_entity.keys())[:args.max_entities])
        sightings_by_entity = {gid: v for gid, v in sightings_by_entity.items() if gid in keep_gids}

    print(f"Sampling up to {args.clips_per_entity} clips each for {len(sightings_by_entity)} entities...")

    pred_path = scene_dir / "pred_full.txt"
    by_cam_frame = defaultdict(list)
    with open(pred_path) as f:
        for line in f:
            cam, gid, frame, l, t, w, h, wx, wy = line.split()
            by_cam_frame[(cam, int(frame))].append((int(gid), float(l), float(t), float(w), float(h)))

    # Reliable sightings only -- same MIN_SIGHTING_DETECTIONS_FOR_CAPTION threshold Phase 1 uses,
    # for the same reason (a short fragment is more likely a track artifact than a real, stable
    # sighting worth describing).
    clip_targets_by_camera = defaultdict(list)  # camera -> [(gid, start_frame, end_frame), ...]
    for gid, sightings in sightings_by_entity.items():
        reliable = [(n, d) for n, d in sightings if d.get("num_detections", 0) >= MIN_SIGHTING_DETECTIONS_FOR_CAPTION]
        source = reliable or sightings
        for cam, start, end in pick_clip_starts(source, args.clip_length, args.clips_per_entity):
            clip_targets_by_camera[cam].append((gid, start, end))

    # Two-phase per camera: resolve every clip's per-frame bboxes first (pure dict lookups), THEN
    # one sequential video pass collecting every needed frame -- see module docstring.
    clips_by_gid = defaultdict(list)  # gid -> [[crop_bgr, ...], ...] (list of clips, each a list of frames)
    for cam, targets in clip_targets_by_camera.items():
        cam_dir = f"camera_{int(cam):04d}"
        cap = cv2.VideoCapture(resolve_video_source(scene_dir / cam_dir))

        # Resolve each clip's per-frame bboxes (only frames where this gid actually has a detection --
        # ByteTrack gaps mean not every frame in [start, end] necessarily has one).
        resolved_clips = []  # [(gid, {frame: (l,t,w,h)}), ...]
        for gid, start, end in targets:
            frame_bboxes = {}
            for fr in range(start, end + 1):
                for cand_gid, l, t, w, h in by_cam_frame.get((cam, fr), []):
                    if cand_gid == gid:
                        frame_bboxes[fr] = (l, t, w, h)
                        break
            if frame_bboxes:
                resolved_clips.append((gid, frame_bboxes))

        targets_by_frame = defaultdict(list)  # frame -> [(gid, clip_idx, l, t, w, h), ...]
        for clip_idx, (gid, frame_bboxes) in enumerate(resolved_clips):
            for fr, (l, t, w, h) in frame_bboxes.items():
                targets_by_frame[fr].append((gid, clip_idx, l, t, w, h))

        clip_frames = defaultdict(dict)  # (gid, clip_idx) -> {frame: crop_bgr}
        if targets_by_frame:
            last_needed_frame = max(targets_by_frame)
            frame_idx = 0
            while frame_idx <= last_needed_frame:
                ok, img = cap.read()
                if not ok:
                    break
                if frame_idx in targets_by_frame:
                    h_img, w_img = img.shape[:2]
                    for gid, clip_idx, l, t, w, h in targets_by_frame[frame_idx]:
                        x1, y1 = max(0, int(l)), max(0, int(t))
                        x2, y2 = min(w_img, int(l + w)), min(h_img, int(t + h))
                        if x2 > x1 and y2 > y1:
                            clip_frames[(gid, clip_idx)][frame_idx] = img[y1:y2, x1:x2]
                frame_idx += 1
        cap.release()

        for (gid, clip_idx), frames_dict in clip_frames.items():
            ordered = [frames_dict[fr] for fr in sorted(frames_dict)]
            clips_by_gid[gid].append(ordered)

    n_clips = sum(len(v) for v in clips_by_gid.values())
    print(f"Extracted {n_clips} clips across {len(clips_by_gid)} entities, running {CAPTION_MODEL_ID} action description...")

    processor = AutoProcessor.from_pretrained(CAPTION_MODEL_ID)
    vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(CAPTION_MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda")
    vlm.eval()
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def describe_clip(frames_bgr: list) -> tuple:
        good_frames = [f for f in frames_bgr if not is_low_quality_crop(f)]
        if len(good_frames) < max(2, len(frames_bgr) * MIN_GOOD_FRAMES_FRACTION):
            return UNCLEAR_ACTION, 1.0  # too many degraded frames to trust -- deterministic
            # decision, not a generation, same convention as extract_entity_attributes.py's
            # crop-quality gate.
        # A person's bbox size shifts slightly frame to frame (walking, turning) -- real bug found
        # here: qwen_vl_utils' fetch_video stacks frames into one tensor and requires identical
        # dimensions, which per-frame bboxes don't naturally give. Resized to the first frame's
        # size (a person's own bbox doesn't change drastically within one ~12-frame/0.4s clip, so
        # this is a mild resize, not a distorting one).
        target_h, target_w = good_frames[0].shape[:2]
        resized = [cv2.resize(f, (target_w, target_h)) for f in good_frames]
        pil_frames = [Image.fromarray(f[:, :, ::-1]) for f in resized]  # BGR -> RGB
        messages = [{"role": "user", "content": [
            {"type": "video", "video": pil_frames, "fps": 1.0},
            {"type": "text", "text": ACTION_CONSTRAINED_PROMPT},
        ]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                            padding=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = vlm.generate(**inputs, max_new_tokens=60, return_dict_in_generate=True, output_scores=True)
        gen = out.sequences[:, inputs["input_ids"].shape[1]:]
        description = processor.batch_decode(gen, skip_special_tokens=True)[0].strip()
        token_probs = [torch.softmax(step_logits[0], dim=-1).max().item() for step_logits in out.scores]
        confidence = sum(token_probs) / len(token_probs) if token_probs else 0.0
        return description, confidence

    n_described = 0
    n_low_agreement = 0
    for gid, clips in clips_by_gid.items():
        descriptions, confidences = [], []
        for frames_bgr in clips:
            desc, conf = describe_clip(frames_bgr)
            descriptions.append(desc)
            confidences.append(conf)

        valid_idxs = [i for i, d in enumerate(descriptions)
                      if not is_degenerate_caption(d) and confidences[i] >= ACTION_CONFIDENCE_THRESHOLD]
        if not valid_idxs:
            valid_idxs = [0]
        valid = [descriptions[i] for i in valid_idxs]
        some_excluded = len(valid) < len(descriptions)

        if len(valid) == 1:
            consensus, agreement = valid[0], (0.0 if some_excluded else 1.0)
        else:
            embs = embedder.encode(valid, normalize_embeddings=True)
            mean_emb = embs.mean(axis=0)
            mean_emb /= np.linalg.norm(mean_emb)
            consensus = valid[int(np.argmax(embs @ mean_emb))]
            n = len(valid)
            pairwise_sum = (embs @ embs.T).sum() - n
            agreement = float(pairwise_sum / (n * (n - 1)))

        G.nodes[f"entity:{gid}"]["action_description"] = consensus
        G.nodes[f"entity:{gid}"]["all_actions"] = descriptions
        G.nodes[f"entity:{gid}"]["all_action_confidences"] = [round(c, 3) for c in confidences]
        G.nodes[f"entity:{gid}"]["action_agreement"] = round(agreement, 3)
        n_described += 1
        if agreement < 0.6:
            n_low_agreement += 1
        if n_described % 100 == 0:
            print(f"  ...{n_described}/{len(clips_by_gid)} ({n_low_agreement} low-agreement so far)")

    print(f"Described {n_described} entities' actions ({n_low_agreement} with action_agreement < 0.6)")

    all_confs = [c for gid in clips_by_gid for c in G.nodes[f"entity:{gid}"]["all_action_confidences"]]
    if all_confs:
        arr = np.array(all_confs)
        pct_excluded = float((arr < ACTION_CONFIDENCE_THRESHOLD).mean() * 100)
        print(f"Action confidence distribution: mean={arr.mean():.3f} median={np.median(arr):.3f} "
              f"p10={np.percentile(arr, 10):.3f} max={arr.max():.3f} -- "
              f"{pct_excluded:.1f}% fall below ACTION_CONFIDENCE_THRESHOLD={ACTION_CONFIDENCE_THRESHOLD}")

    with open(args.out, "wb") as f:
        pickle.dump(G, f)
    print(f"Saved graph with actions to {args.out}")


if __name__ == "__main__":
    main()
