"""
Per-camera step 1 of cross-camera ReID: for each local track (from
track_single_camera.py's tracks.jsonl), sample a few frames, crop the
person, and compute an appearance embedding with a Market1501-finetuned
OSNet (see ../CLAUDE.md's "ReID model chosen" note for why this specific
checkpoint). Averages each track's sampled-frame embeddings into one
L2-normalized vector per track -- robust to a single bad crop (motion
blur, partial occlusion) without needing per-frame identity decisions.

Does ONE sequential pass over the video (only decoding frames that are
actually needed) rather than random-seeking per sample, since OpenCV
seeks on compressed video are slow and 24k-frame videos make random
access the wrong default.

Usage:
  python scripts/extract_track_embeddings.py --camera data/scene_061/camera_0535
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from crop_quality import is_low_quality_crop
from geometry import load_homography, bbox_ground_point, image_to_world
from video_source import resolve_video_source


def pick_sample_frames(frames: list, k: int) -> set:
    """k evenly-spaced frames from a track's (sorted) frame list, dedup'd."""
    if len(frames) <= k:
        return set(frames)
    idxs = np.linspace(0, len(frames) - 1, k).round().astype(int)
    return {frames[i] for i in idxs}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=str, required=True)
    parser.add_argument("--samples_per_track", type=int, default=5)
    parser.add_argument("--reid_checkpoint", type=str,
                         default=str(PROJECT_ROOT / "checkpoints" / "osnet_x1_0_market1501.pth"))
    parser.add_argument("--out", type=str, default=None,
                         help="Output .npz path; default: <camera>/track_embeddings.npz")
    args = parser.parse_args()

    camera_dir = Path(args.camera)
    tracks_path = camera_dir / "tracks.jsonl"
    out_path = Path(args.out) if args.out else camera_dir / "track_embeddings.npz"

    # Load detections, keep only person class (this is a people-ReID model).
    dets_by_frame = defaultdict(list)  # frame -> [(track_id, bbox_left, top, w, h)]
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
    target_frames = defaultdict(list)  # frame -> [track_id, ...] to crop at this frame
    for t, fset in sample_frames_by_track.items():
        for fr in fset:
            target_frames[fr].append(t)

    print(f"{len(frames_by_track)} tracks, sampling up to {args.samples_per_track} frames each "
          f"-> {sum(len(v) for v in target_frames.values())} crops to extract from {len(target_frames)} frames")

    # Single sequential pass: grab crops only at target frames.
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
                    crop_bgr = frame[y1:y2, x1:x2]
                    if is_low_quality_crop(crop_bgr):
                        continue  # too small/blurry to trust for a ReID embedding -- same gate
                        # used before captioning (crop_quality.py), applied here because a
                        # spot-check during Phase 2's within-camera stitching sweep found
                        # degenerate crops clustering together in embedding space and getting
                        # wrongly merged as "the same person" purely on shared noise, not real
                        # appearance similarity.
                    # cv2 reads BGR; torchreid's FeatureExtractor treats numpy arrays as RGB
                    # (via ToPILImage, no channel-order conversion) -- must flip here or every
                    # crop's color signal (the primary ReID cue) is systematically wrong.
                    crops_by_track[tid].append(crop_bgr[:, :, ::-1])
        frame_idx += 1
    cap.release()
    n_dropped_all_low_quality = sum(1 for tid in frames_by_track if tid not in crops_by_track)
    print(f"Read {frame_idx} frames, collected crops for {len(crops_by_track)} tracks "
          f"({n_dropped_all_low_quality} tracks had every sampled crop filtered as low-quality, "
          f"no embedding -- excluded rather than embedding pure noise)")

    from torchreid.reid.utils import FeatureExtractor
    extractor = FeatureExtractor(model_name="osnet_x1_0", model_path=args.reid_checkpoint, device="cuda")

    track_ids, embeddings = [], []
    homography = load_homography(camera_dir / "calibration.json")
    world_positions = []
    first_world_positions, last_world_positions = [], []
    first_pixel_positions, last_pixel_positions = [], []
    for tid, crops in crops_by_track.items():
        feats = extractor(crops).cpu().numpy()  # (n_crops, 512)
        feats = feats / np.linalg.norm(feats, axis=1, keepdims=True)
        mean_feat = feats.mean(axis=0)
        mean_feat = mean_feat / np.linalg.norm(mean_feat)
        track_ids.append(tid)
        embeddings.append(mean_feat)

        # Median world position across the track's own full detection history (not just samples)
        # -- gives cross-camera matching a spatial signal alongside appearance.
        positions = []
        pixel_positions = []
        for fr in frames_by_track[tid]:
            for t2, l, t, w, h in dets_by_frame[fr]:
                if t2 == tid:
                    gx, gy = bbox_ground_point(l, t, w, h)
                    positions.append(image_to_world(gx, gy, homography))
                    pixel_positions.append((gx, gy))
        positions = np.array(positions)
        world_positions.append(np.median(positions, axis=0))
        # First/last-frame position specifically (not the median) -- real bug found downstream in
        # cross_camera_reid.py's within-camera stitching: a same-camera "clean handoff" check (does
        # track A's END position sit right where track B's START position is, moments later) needs
        # the ACTUAL boundary-frame position, not the track-wide median. For a long, wandering track
        # the median can sit far from either endpoint. Tried WORLD-coordinate endpoints first, but
        # found (also empirically, comparing against known-good visually-confirmed handoff pairs)
        # that homography-projected world distance stayed large (100s of units) even for genuine
        # clean handoffs -- ground-plane projection accuracy degrades away from the camera, and
        # apparently amplifies small pixel differences enough to make world-distance unreliable for
        # THIS specific tight-tolerance check (unlike the coarser walking-speed formula, which
        # tolerates the noise fine). Pixel-space endpoint distance is used instead for the handoff
        # check -- a much more direct, less error-amplified measure of "is this the same detection
        # continuing," confirmed against the same real pairs (single-digit-to-20px, vs. 100s of
        # units in world space for the same pairs).
        first_world_positions.append(positions[0])
        last_world_positions.append(positions[-1])
        first_pixel_positions.append(pixel_positions[0])
        last_pixel_positions.append(pixel_positions[-1])

    embeddings = np.stack(embeddings) if embeddings else np.zeros((0, 512))
    world_positions = np.stack(world_positions) if world_positions else np.zeros((0, 2))
    first_world_positions = np.stack(first_world_positions) if first_world_positions else np.zeros((0, 2))
    last_world_positions = np.stack(last_world_positions) if last_world_positions else np.zeros((0, 2))
    first_pixel_positions = np.stack(first_pixel_positions) if first_pixel_positions else np.zeros((0, 2))
    last_pixel_positions = np.stack(last_pixel_positions) if last_pixel_positions else np.zeros((0, 2))
    first_frames = np.array([frames_by_track[t][0] for t in track_ids])
    last_frames = np.array([frames_by_track[t][-1] for t in track_ids])

    np.savez(out_path, track_ids=np.array(track_ids), embeddings=embeddings,
             world_positions=world_positions, first_frames=first_frames, last_frames=last_frames,
             first_pixel_positions=first_pixel_positions, last_pixel_positions=last_pixel_positions,
             first_world_positions=first_world_positions, last_world_positions=last_world_positions)
    print(f"Saved {len(track_ids)} track embeddings to {out_path}")


if __name__ == "__main__":
    main()
