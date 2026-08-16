"""
Per-camera step, parallel to extract_track_embeddings.py: for each LOCAL
(pre-merge) track, sample a couple of crops, caption them with the same
constrained-prompt Qwen2.5-VL-3B setup as extract_entity_attributes.py,
and store a sentence-embedding of the resulting text per track.

Exists because captions currently only get computed AFTER the event graph
is built (post cross_camera_reid.py merging) -- too late to use as a THIRD
matching signal (appearance embedding + world position + now caption text)
during identity resolution itself. Per user direction: match identities by
appearance, wearing (caption), and position -- not appearance alone.

Kept deliberately simpler than extract_entity_attributes.py: no multi-crop
medoid/agreement consensus (that machinery exists for the QUERY-facing,
human-readable appearance_caption field) -- here the caption text is only
ever consumed as an embedding for a similarity score, so a single
representative crop's caption (or the concatenation of 2) is enough. Still
reuses the SAME constrained prompt, color splicing, and crop-quality gate
as Phase 1, since a garbage caption from a degenerate crop would poison the
similarity signal the same way it would poison a query-facing caption.

Usage:
  python scripts/extract_track_captions.py --camera data/scene_061/camera_0535
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from color_utils import garment_region, dominant_color_name, strip_color_words
from crop_quality import is_low_quality_crop, UNCLEAR_CAPTION
from video_source import resolve_video_source

CAPTION_MODEL_ID = "Qwen/Qwen2.5-VL-3B-Instruct"
SAMPLES_PER_TRACK = 2  # fewer than extract_track_embeddings.py's 5 -- this is a coarser signal,
# and captioning is far more expensive per-crop than a ReID embedding forward pass.

CONSTRAINED_PROMPT = (
    "Describe ONLY the visible clothing type and accessories of the person in this image "
    "(e.g. jacket, t-shirt, backpack, hat). Do NOT mention color. Do NOT describe any action, "
    "activity, pose, or what the person might be doing. If something is unclear or not visible, "
    "say so briefly instead of guessing."
)  # same prompt validated in the Phase 0 bake-off / used by extract_entity_attributes.py


def pick_sample_frames(frames: list, k: int) -> set:
    if len(frames) <= k:
        return set(frames)
    idxs = np.linspace(0, len(frames) - 1, k).round().astype(int)
    return {frames[i] for i in idxs}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=str, required=True)
    parser.add_argument("--samples_per_track", type=int, default=SAMPLES_PER_TRACK)
    parser.add_argument("--out", type=str, default=None,
                         help="Output .npz path; default: <camera>/track_captions.npz")
    args = parser.parse_args()

    camera_dir = Path(args.camera)
    tracks_path = camera_dir / "tracks.jsonl"
    out_path = Path(args.out) if args.out else camera_dir / "track_captions.npz"

    dets_by_frame = defaultdict(list)
    frames_by_track = defaultdict(list)
    with open(tracks_path) as f:
        for line in f:
            d = json.loads(line)
            if d["class"] != "person":
                continue
            dets_by_frame[d["frame"]].append((d["track_id"], d["bbox_left"], d["bbox_top"], d["bbox_width"], d["bbox_height"]))
            frames_by_track[d["track_id"]].append(d["frame"])
    for t in frames_by_track:
        frames_by_track[t].sort()

    sample_frames_by_track = {t: pick_sample_frames(fs, args.samples_per_track) for t, fs in frames_by_track.items()}
    target_frames = defaultdict(list)
    for t, fset in sample_frames_by_track.items():
        for fr in fset:
            target_frames[fr].append(t)

    print(f"{len(frames_by_track)} tracks, sampling up to {args.samples_per_track} crops each "
          f"-> {sum(len(v) for v in target_frames.values())} crops to extract from {len(target_frames)} frames")

    crops_by_track = defaultdict(list)
    cap = cv2.VideoCapture(resolve_video_source(camera_dir))
    frame_idx = 0
    det_lookup = {fr: {tid: (l, t, w, h) for tid, l, t, w, h in dets} for fr, dets in dets_by_frame.items()}
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_idx in target_frames:
            h_img, w_img = frame.shape[:2]
            for tid in target_frames[frame_idx]:
                l, t, w, h = det_lookup[frame_idx][tid]
                x1, y1 = max(0, int(l)), max(0, int(t))
                x2, y2 = min(w_img, int(l + w)), min(h_img, int(t + h))
                if x2 > x1 and y2 > y1:
                    crops_by_track[tid].append(frame[y1:y2, x1:x2])  # BGR, native
        frame_idx += 1
    cap.release()
    print(f"Read {frame_idx} frames, collected crops for {len(crops_by_track)} tracks")

    processor = AutoProcessor.from_pretrained(CAPTION_MODEL_ID)
    vlm = Qwen2_5_VLForConditionalGeneration.from_pretrained(CAPTION_MODEL_ID, torch_dtype=torch.bfloat16, device_map="cuda")
    vlm.eval()
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    def caption_one(crop_bgr: np.ndarray) -> str:
        if is_low_quality_crop(crop_bgr):
            return UNCLEAR_CAPTION
        upper = dominant_color_name(garment_region(crop_bgr, "upper"))
        lower = dominant_color_name(garment_region(crop_bgr, "lower"))
        image = Image.fromarray(crop_bgr[:, :, ::-1])
        messages = [{"role": "user", "content": [{"type": "image", "image": image},
                                                   {"type": "text", "text": CONSTRAINED_PROMPT}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=[image], return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = vlm.generate(**inputs, max_new_tokens=60)
        gen = out[:, inputs["input_ids"].shape[1]:]
        garment_text = strip_color_words(processor.batch_decode(gen, skip_special_tokens=True)[0].strip())
        return f"{upper} top, {lower} bottom. {garment_text}"

    track_ids, captions = [], []
    n_unclear = 0
    for tid, crops in crops_by_track.items():
        texts = [caption_one(c) for c in crops]
        valid = [t for t in texts if not t.startswith(UNCLEAR_CAPTION)]
        caption = " ".join(valid) if valid else texts[0]
        if not valid:
            n_unclear += 1
        track_ids.append(tid)
        captions.append(caption)

    embeddings = embedder.encode(captions, normalize_embeddings=True) if captions else np.zeros((0, 384))

    np.savez(out_path, track_ids=np.array(track_ids), caption_embeddings=embeddings,
             captions=np.array(captions, dtype=object))
    print(f"Saved {len(track_ids)} track captions to {out_path} "
          f"({n_unclear} tracks had every sampled crop unclear)")


if __name__ == "__main__":
    main()
