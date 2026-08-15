"""
Download a small working subset of nvidia/PhysicalAI-SmartSpaces
(MTMC_Tracking_2024, test split): one scene's ground_truth.txt plus a
handful of its cameras' video.mp4 + calibration.json.

See ../CLAUDE.md for dataset background, ethics rationale, and the
ground_truth.txt column format.

Usage:
  python scripts/download_data.py
  python scripts/download_data.py --scene scene_061 --cameras 0535 0536 0537 0538
"""
import argparse
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REPO_ID = "nvidia/PhysicalAI-SmartSpaces"
EDITION = "MTMC_Tracking_2024"
SPLIT = "test"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene", type=str, default="scene_061")
    parser.add_argument("--cameras", type=str, nargs="+", default=["0535", "0536", "0537", "0538"],
                         help="4-digit camera IDs (without the 'camera_' prefix)")
    parser.add_argument("--out_dir", type=str, default=str(PROJECT_ROOT / "data"))
    args = parser.parse_args()

    from huggingface_hub import hf_hub_download

    out_dir = Path(args.out_dir) / args.scene
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading ground_truth.txt for {args.scene}...")
    gt_path = hf_hub_download(
        repo_id=REPO_ID, repo_type="dataset",
        filename=f"{EDITION}/{SPLIT}/{args.scene}/ground_truth.txt",
    )
    shutil.copy(gt_path, out_dir / "ground_truth.txt")

    for cam in args.cameras:
        cam_name = f"camera_{cam}"
        cam_out = out_dir / cam_name
        cam_out.mkdir(exist_ok=True)

        print(f"Downloading {cam_name}/calibration.json...")
        cal_path = hf_hub_download(
            repo_id=REPO_ID, repo_type="dataset",
            filename=f"{EDITION}/{SPLIT}/{args.scene}/{cam_name}/calibration.json",
        )
        shutil.copy(cal_path, cam_out / "calibration.json")

        print(f"Downloading {cam_name}/video.mp4 (~155MB, this is the slow part)...")
        video_path = hf_hub_download(
            repo_id=REPO_ID, repo_type="dataset",
            filename=f"{EDITION}/{SPLIT}/{args.scene}/{cam_name}/video.mp4",
        )
        shutil.copy(video_path, cam_out / "video.mp4")

    print(f"\nDone. Data for {args.scene} ({len(args.cameras)} cameras) saved to {out_dir}")


if __name__ == "__main__":
    main()
