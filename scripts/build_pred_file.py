"""
Build a prediction file in the official 9-column format (camera_id
object_id frame_id bbox_left bbox_top bbox_width bbox_height world_x
world_y) from one or more cameras' tracks.jsonl, for scoring with the
vendored TrackEval code in ../eval_ref.

IMPORTANT: track_id from track_single_camera.py is only unique WITHIN a
camera (ByteTrack has no notion of other cameras). Passing multiple
cameras here WITHOUT --global_id_map will produce a prediction file
where e.g. "object 1" in camera_0535 and "object 1" in camera_0536 are
almost certainly different real people -- TrackEval will score that as
if they were the same object, which is wrong for a genuine multi-camera
evaluation. Either use --cameras with a SINGLE camera (evaluates
detection + single-camera tracking in isolation), or pass
--global_id_map from cross_camera_reid.py to remap local track_ids to
cross-camera-consistent global ids first.

Usage:
  python scripts/build_pred_file.py --scene_dir data/scene_061 --cameras 0535 --out data/scene_061/pred_cam0535.txt
  python scripts/build_pred_file.py --scene_dir data/scene_061 --cameras 0535 0536 0537 0538 \
      --global_id_map data/scene_061/global_id_map.json --out data/scene_061/pred_full.txt
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import json
from geometry import load_homography, bbox_ground_point, image_to_world


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_dir", type=str, required=True)
    parser.add_argument("--cameras", type=str, nargs="+", required=True,
                         help="4-digit camera IDs (without 'camera_' prefix)")
    parser.add_argument("--global_id_map", type=str, default=None,
                         help="JSON from cross_camera_reid.py mapping 'cam,track_id' -> global_id")
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    scene_dir = Path(args.scene_dir)
    rows_written = 0

    global_id_map = None
    if args.global_id_map:
        with open(args.global_id_map) as f:
            global_id_map = json.load(f)

    with open(args.out, "w") as out_f:
        for cam in args.cameras:
            cam_dir = scene_dir / f"camera_{cam}"
            tracks_path = cam_dir / "tracks.jsonl"
            homography = load_homography(cam_dir / "calibration.json")

            with open(tracks_path) as f:
                for line in f:
                    d = json.loads(line)
                    if d["class"] != "person":
                        # ground_truth.txt has no class column -- this benchmark is
                        # people-only ("Multi-Camera People Tracking", per eval/main.py's
                        # docstring), so vehicle detections aren't evaluable here.
                        continue
                    out_id = d["track_id"]
                    if global_id_map is not None:
                        key = f"{cam},{d['track_id']}"
                        if key not in global_id_map:
                            continue  # track had no sampled crops (e.g. all-invalid bboxes); skip
                        out_id = global_id_map[key]
                    gx, gy = bbox_ground_point(d["bbox_left"], d["bbox_top"], d["bbox_width"], d["bbox_height"])
                    wx, wy = image_to_world(gx, gy, homography)
                    out_f.write(
                        f"{int(cam)} {out_id} {d['frame']} "
                        f"{d['bbox_left']:.2f} {d['bbox_top']:.2f} {d['bbox_width']:.2f} {d['bbox_height']:.2f} "
                        f"{wx:.4f} {wy:.4f}\n"
                    )
                    rows_written += 1

    print(f"Wrote {rows_written} prediction rows to {args.out}")


if __name__ == "__main__":
    main()
