"""
Stage: per-camera detection + tracking. Runs a pretrained YOLO detector +
ByteTrack (both via ultralytics) over one camera's video, producing
per-frame tracklets: (frame_id, track_id, class, bbox).

This is OUR tracking output -- separate from ground_truth.txt's object_id,
which is the dataset's ground truth. Comparing the two (next script) is
how we quantitatively evaluate tracking quality instead of guessing.

Classes: COCO-pretrained YOLO only knows COCO's 80 classes. We keep
person (0) and vehicle-ish classes (car=2, motorcycle=3, bus=5, truck=7)
since those are the entities this dataset's queries would care about --
NOT the full 80-class list, to keep output focused and fast.

Usage:
  python scripts/track_single_camera.py --camera data/scene_061/camera_0535 --max_frames 900
  python scripts/track_single_camera.py --camera data/scene_061/camera_0535   # full video
"""
import argparse
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
import reid_rect_patch  # noqa: F401 -- must be imported before any BoT-SORT+ReID tracker is
# constructed; patches ultralytics' square-only, unnormalized ReID preprocessing to match this
# project's rectangular, ImageNet-normalized OSNet convention. A no-op for ByteTrack/ReID-less runs.
from video_source import resolve_track_source

# COCO class ids we care about for this project's entity/event vocabulary.
CLASSES_OF_INTEREST = {0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=str, required=True,
                         help="Path to a camera dir, e.g. data/scene_061/camera_0535")
    parser.add_argument("--model", type=str, default="yolo11s.pt",
                         help="yolo11s chosen over yolo11n/yolo11m via a HOTA sweep on camera_0535: "
                              "n@conf0.3=28.06 HOTA, n@conf0.15=27.81, s@conf0.15=33.58 (best), "
                              "m@conf0.15=32.72 (no better than s, not worth the extra cost)")
    parser.add_argument("--conf", type=float, default=0.15)
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml",
                         help="Tracker config; use bytetrack_tuned.yaml (track_buffer=90) for crowded cameras")
    parser.add_argument("--max_frames", type=int, default=None,
                         help="Limit to first N frames (smoke test); default: full video")
    parser.add_argument("--out", type=str, default=None,
                         help="Output JSONL path; default: <camera>/tracks.jsonl")
    args = parser.parse_args()

    from ultralytics import YOLO

    camera_dir = Path(args.camera)
    video_path = resolve_track_source(camera_dir)
    out_path = Path(args.out) if args.out else camera_dir / "tracks.jsonl"

    model = YOLO(args.model)

    print(f"Tracking {video_path} (classes={list(CLASSES_OF_INTEREST.values())}, "
          f"max_frames={args.max_frames or 'all'})...")

    results = model.track(
        source=str(video_path),
        classes=list(CLASSES_OF_INTEREST.keys()),
        conf=args.conf,
        tracker=args.tracker,
        persist=True,
        stream=True,
        verbose=False,
    )

    written = 0
    with open(out_path, "w") as f:
        for frame_idx, r in enumerate(results):
            if args.max_frames is not None and frame_idx >= args.max_frames:
                break
            if r.boxes is None or r.boxes.id is None:
                continue
            boxes_xywh = r.boxes.xywh.cpu().numpy()
            track_ids = r.boxes.id.cpu().numpy().astype(int)
            cls_ids = r.boxes.cls.cpu().numpy().astype(int)
            confs = r.boxes.conf.cpu().numpy()
            for (cx, cy, w, h), tid, cls_id, conf in zip(boxes_xywh, track_ids, cls_ids, confs):
                f.write(json.dumps({
                    "frame": frame_idx,
                    "track_id": int(tid),
                    "class": CLASSES_OF_INTEREST[int(cls_id)],
                    "bbox_left": float(cx - w / 2),
                    "bbox_top": float(cy - h / 2),
                    "bbox_width": float(w),
                    "bbox_height": float(h),
                    "conf": float(conf),
                }) + "\n")
                written += 1
            if (frame_idx + 1) % 300 == 0:
                print(f"  ...frame {frame_idx + 1}, {written} detections so far")

    print(f"Done. {written} detections written to {out_path}")


if __name__ == "__main__":
    main()
