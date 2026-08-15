"""
Score a prediction file against ground_truth.txt using the vendored
official TrackEval code (../eval_ref, copied from the dataset repo's own
eval/ directory -- see ../CLAUDE.md's "Official evaluation code" section
for why we reuse this instead of a hand-rolled metric).

Filters ground_truth.txt down to just the given cameras (so e.g. a
single-camera tracking-quality check isn't penalized for cameras we
never downloaded/ran detection on), writes the scene_2_camera_id.json
the official script requires, then calls its evaluate() function
directly.

Usage:
  # single-camera tracking-quality check (no cross-camera ID needed):
  python scripts/evaluate_tracking.py --scene_dir data/scene_061 --cameras 0535 \
      --pred data/scene_061/pred_cam0535.txt --scene_name scene_061_cam0535

  # full 4-camera cross-camera check (once ReID assigns global IDs):
  python scripts/evaluate_tracking.py --scene_dir data/scene_061 --cameras 0535 0536 0537 0538 \
      --pred data/scene_061/pred_full.txt --scene_name scene_061
"""
import argparse
import json
import sys
import tempfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
EVAL_REF = PROJECT_ROOT / "eval_ref"
sys.path.insert(0, str(EVAL_REF))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_dir", type=str, required=True)
    parser.add_argument("--cameras", type=str, nargs="+", required=True)
    parser.add_argument("--pred", type=str, required=True)
    parser.add_argument("--scene_name", type=str, required=True)
    parser.add_argument("--num_cores", type=int, default=4)
    args = parser.parse_args()

    import main as official_eval  # eval_ref/main.py

    scene_dir = Path(args.scene_dir)
    camera_ids = {int(c) for c in args.cameras}

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # Filter the full scene ground truth down to just our cameras.
        gt_filtered_path = tmp / "ground_truth.txt"
        with open(scene_dir / "ground_truth.txt") as src, open(gt_filtered_path, "w") as dst:
            for line in src:
                cam_id = int(line.split()[0])
                if cam_id in camera_ids:
                    dst.write(line)

        scene_map_path = tmp / "scene_2_camera_id.json"
        with open(scene_map_path, "w") as f:
            json.dump([{"scene_name": args.scene_name, "camera_ids": sorted(camera_ids)}], f)

        result = official_eval.evaluate(
            prediction_file=args.pred,
            ground_truth_file=str(gt_filtered_path),
            output_dir=None,
            num_cores=args.num_cores,
            scene_2_camera_id_file=str(scene_map_path),
        )

    print(f"\n=== Results for {args.scene_name} (cameras {sorted(camera_ids)}) ===")
    print(f"HOTA: {float(result['FINAL']['HOTA'] * 100):.4f}%")
    print(f"DetA: {float(result['FINAL']['DetA'] * 100):.4f}%")
    print(f"AssA: {float(result['FINAL']['AssA'] * 100):.4f}%")
    print(f"LocA: {float(result['FINAL']['LocA'] * 100):.4f}%")


if __name__ == "__main__":
    main()
