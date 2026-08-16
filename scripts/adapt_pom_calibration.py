"""
Calibration adapter for the EPFL CVLab "POM" (Probabilistic Occupancy Map)
multi-camera pedestrian dataset -- converts its native calibration formats
into this project's existing calibration.json shape ({"homography matrix":
[[...]]}), so the rest of the pipeline (geometry.py, build_pred_file.py,
extract_track_embeddings.py, cross_camera_reid.py) needs ZERO changes:
confirmed by grep that nothing downstream reads any OTHER calibration.json
field. See CLAUDE.md for the full multi-source-data decision writeup.

POM ships calibration in TWO independent forms for the same cameras:
1. calibration-terrace.txt: a directly-given 3x3 ground-plane homography
   per camera, plus a "head plane height" scalar this project deliberately
   does not use (bottom-center ground-contact projection is the
   established convention here, see geometry.py, not a head-plane model).
2. terrace-tsai.zip: a classic Tsai camera model (intrinsics + distortion
   + extrinsics) for the SAME 4 cameras -- an independent calibration this
   script uses ONLY to verify the given homography's direction (does it
   map world->image like geometry.py already assumes, or the reverse?),
   since there's no ground_truth.txt-equivalent bbox list to check against
   the way scene_061's original calibration direction was confirmed.

Two things to watch, verified rather than assumed here:
- The Tsai XMLs' Geometry says width=720/height=576, but the actual
  downloaded terrace1-c*.avi videos are 360x288 (confirmed via cv2) --
  exactly half in both dimensions. The Tsai-derived homography is
  computed at native (720x576) scale; comparison against the given
  homography (whose own native scale is unknown) is tried at both native
  and half scale, and the result is reported explicitly, not assumed.
- The Tsai rx/ry/rz triplet is assumed to be a Rodrigues rotation vector
  (axis-angle, decoded via cv2.Rodrigues) -- the common convention for
  this exact toolkit lineage (this dataset and EPFL-RLC share the same
  XML schema), but unverified beyond that assumption holding up in the
  cross-check below.

Usage:
  python scripts/adapt_pom_calibration.py --scene_dir data/POM/terrace1
"""
import argparse
import json
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_given_homographies(calibration_txt: Path) -> dict:
    """Parses calibration-terrace.txt's per-camera 3x3 "Ground plane
    homography" blocks. Returns {camera_index: 3x3 np.ndarray}."""
    text = calibration_txt.read_text()
    homographies = {}
    blocks = text.split("# Camera ")[1:]
    for block in blocks:
        # First line of the block is the camera index (e.g. "0\n##########...").
        cam_idx = int(block.splitlines()[0].strip())
        lines_after_marker = block.split("# Ground plane homography")[1]
        numeric_lines = [ln for ln in lines_after_marker.splitlines() if ln.strip() and not ln.strip().startswith("#")]
        rows = []
        for ln in numeric_lines[:3]:
            rows.append([float(x) for x in ln.split()])
        homographies[cam_idx] = np.array(rows, dtype=np.float64)
    return homographies


def parse_tsai_xml(xml_path: Path) -> dict:
    root = ET.parse(xml_path).getroot()
    geom = root.find("Geometry")
    intr = root.find("Intrinsic")
    extr = root.find("Extrinsic")
    return {
        "width": float(geom.get("width")), "height": float(geom.get("height")),
        "focal": float(intr.get("focal")), "kappa1": float(intr.get("kappa1")),
        "cx": float(intr.get("cx")), "cy": float(intr.get("cy")), "sx": float(intr.get("sx")),
        "dpx": float(geom.get("dpx")), "dpy": float(geom.get("dpy")),
        "tx": float(extr.get("tx")), "ty": float(extr.get("ty")), "tz": float(extr.get("tz")),
        "rx": float(extr.get("rx")), "ry": float(extr.get("ry")), "rz": float(extr.get("rz")),
    }


def tsai_ground_plane_homography(tsai: dict) -> np.ndarray:
    """Derives a WORLD->IMAGE ground-plane homography from a Tsai camera model,
    at the calibration's own native image resolution -- the same "drop the
    Z-column from P = K[R|t]" technique already used (and empirically
    confirmed) for scene_061's own calibration.json. Ignores radial
    distortion (kappa1): this is only used as an independent direction/scale
    CHECK against a linear (homography) model, and distortion isn't linear
    anyway, so the undistorted pinhole approximation is what's comparable."""
    fx = tsai["focal"] * tsai["sx"] / tsai["dpx"]
    fy = tsai["focal"] / tsai["dpy"]
    K = np.array([[fx, 0, tsai["cx"]], [0, fy, tsai["cy"]], [0, 0, 1]])
    rvec = np.array([tsai["rx"], tsai["ry"], tsai["rz"]])
    R, _ = cv2.Rodrigues(rvec)
    t = np.array([tsai["tx"], tsai["ty"], tsai["tz"]])
    P = K @ np.column_stack([R, t])  # 3x4
    H = P[:, [0, 1, 3]]  # drop the Z column -- ground-plane (Z=0) homography
    return H


def normalize(H: np.ndarray) -> np.ndarray:
    """Homographies are only defined up to scale -- normalize by the bottom-right
    element (standard convention) so two matrices representing the same mapping
    are directly, numerically comparable."""
    return H / H[2, 2] if abs(H[2, 2]) > 1e-12 else H / np.linalg.norm(H)


def compare(name: str, a: np.ndarray, b: np.ndarray) -> float:
    """Relative Frobenius-norm difference between two normalized homographies --
    near 0 means "these represent the same mapping"."""
    a, b = normalize(a), normalize(b)
    diff = np.linalg.norm(a - b) / np.linalg.norm(b)
    print(f"  {name}: relative diff = {diff:.4f}")
    return diff


def verify_via_detections(scene_dir: Path, given: dict, num_cameras: int) -> str:
    """The actual DECISIVE verification (the Tsai cross-derivation above turned
    out too numerically unreliable to trust on its own, likely from a wrong
    convention assumption -- see this function's usage in main() for the
    honest account of that). Runs real YOLO person detections on one frame per
    camera, maps each detection's bottom-center through BOTH candidate uses of
    the given homography, and checks which one makes DIFFERENT cameras'
    detections converge on the SAME real-world positions -- since all cameras
    observe the same physical terrace, genuine world coordinates from
    different cameras should cluster together; a wrong direction/scale
    produces no such agreement. Returns "store_as_is" or "store_inverted"."""
    from ultralytics import YOLO
    model = YOLO(str(PROJECT_ROOT / "yolo11s.pt"))

    def to_world(H, x, y):
        p = H @ np.array([x, y, 1.0])
        return np.array([p[0] / p[2], p[1] / p[2]])

    results_by_case = {"store_as_is": [], "store_inverted": []}  # each: [(cam, point), ...]
    for cam in range(num_cameras):
        cap = cv2.VideoCapture(str(scene_dir / f"terrace1-c{cam}.avi"))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 500)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            continue
        detections = model.predict(frame, classes=[0], conf=0.3, verbose=False)[0]
        H = given[cam]
        for box in detections.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            gx, gy = (x1 + x2) / 2, y2  # bbox bottom-center, ground-contact point
            # "store_as_is": calibration.json keeps H unmodified -> geometry.py
            # computes inv(H) @ point internally -- reproduce that here directly.
            results_by_case["store_as_is"].append((cam, to_world(np.linalg.inv(H), gx, gy)))
            # "store_inverted": calibration.json would store inv(H) -> geometry.py
            # computes inv(inv(H)) @ point = H @ point -- reproduce that here.
            results_by_case["store_inverted"].append((cam, to_world(H, gx, gy)))

    # Score = mean distance from each detection to its NEAREST detection from a
    # DIFFERENT camera (not the overall mean pairwise distance across all
    # points, which was tried first and found misleading: it conflates
    # "different real people" -- expected to be far apart regardless of
    # calibration correctness -- with "the same real person seen by two
    # cameras" -- which should collapse to ~0 distance under a correct
    # calibration. Restricting to cross-camera nearest-neighbor isolates the
    # signal that actually indicates calibration correctness.
    scores = {}
    for case, entries in results_by_case.items():
        if len(entries) < 2:
            continue
        nearest_cross_cam_dists = []
        for i, (cam_i, pt_i) in enumerate(entries):
            others = [pt_j for j, (cam_j, pt_j) in enumerate(entries) if cam_j != cam_i]
            if others:
                nearest_cross_cam_dists.append(min(np.linalg.norm(pt_i - pt_j) for pt_j in others))
        scores[case] = float(np.mean(nearest_cross_cam_dists))
        print(f"  {case}: {len(entries)} detections, mean nearest-cross-camera-detection "
              f"world-distance = {scores[case]:.1f}")

    winner = min(scores, key=scores.get)
    print(f"  -> tighter cross-camera agreement (a correct calibration should have near-0 "
          f"distance between the SAME real person seen by different cameras): {winner}")
    return winner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_dir", type=str, required=True)
    parser.add_argument("--num_cameras", type=int, default=4)
    args = parser.parse_args()
    scene_dir = Path(args.scene_dir)

    given = parse_given_homographies(scene_dir / "calibration-terrace.txt")

    tsai_dir = scene_dir / "tsai"
    if not tsai_dir.exists():
        tsai_dir.mkdir()
        with zipfile.ZipFile(scene_dir / "terrace-tsai.zip") as z:
            z.extractall(tsai_dir)

    print("Cross-checking against an independently Tsai-derived homography (diagnostic only):")
    for cam in range(args.num_cameras):
        tsai = parse_tsai_xml(tsai_dir / f"terrace-tsai-c{cam}.xml")
        H_tsai_native = tsai_ground_plane_homography(tsai)  # at 720x576

        # Actual downloaded video resolution vs. the Tsai calibration's native
        # resolution -- checked, not assumed, via cv2 on the real video file.
        cap = cv2.VideoCapture(str(scene_dir / f"terrace1-c{cam}.avi"))
        video_w, video_h = cap.get(cv2.CAP_PROP_FRAME_WIDTH), cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        cap.release()
        scale = video_w / tsai["width"]  # e.g. 360/720 = 0.5

        # A uniform image-scale-by-`scale` composes with a homography as
        # diag(scale,scale,1) @ H -- scales the first two ROWS of the image-side
        # output, which is what a downsampled-resolution image needs.
        S = np.diag([scale, scale, 1.0])
        H_tsai_scaled = S @ H_tsai_native

        print(f"camera {cam} (video {video_w:.0f}x{video_h:.0f}, "
              f"tsai native {tsai['width']:.0f}x{tsai['height']:.0f}, scale={scale}):")
        H_given = given[cam]
        d_direct = compare("given vs tsai(scaled)", H_given, H_tsai_scaled)
        d_inv = compare("given^-1 vs tsai(scaled)", np.linalg.inv(H_given), H_tsai_scaled)
        d_native = compare("given vs tsai(native, no scale)", H_given, H_tsai_native)

        best = min([("given_as_world_to_image", d_direct),
                    ("given_needs_inverting", d_inv),
                    ("given_at_native_res_no_scale", d_native)], key=lambda x: x[1])
        print(f"  -> closest (unreliable) match: {best[0]} (diff={best[1]:.4f})\n")

    # The Tsai cross-derivation above is NOT trustworthy on its own: every
    # camera's best-case residual is 90-300% (checked, not glossed over) --
    # far too large to represent a real match, most likely from a wrong
    # assumption in reconstructing the Tsai model here (rotation convention,
    # focal/pixel-unit conversion, or similar -- genuinely ambiguous without
    # the original toolkit's exact documented convention). It's kept as
    # printed diagnostic output, not used to decide anything.
    print("(Tsai cross-check was inconclusive -- large residuals in every camera, not used "
          "to decide the direction. Deciding via real multi-camera detection consistency instead.)\n")

    print("Verifying homography direction via multi-camera detection consistency:")
    decision = verify_via_detections(scene_dir, given, args.num_cameras)
    # decision == "store_as_is" means: keep the given homography exactly as
    # parsed. geometry.py always inverts whatever calibration.json stores, so
    # storing it unmodified makes geometry.py compute inv(given) @ image_point
    # -- confirmed empirically above to be the case that makes different
    # cameras' detections of the same real people converge on the same world
    # coordinates, not the case that scatters them.

    for cam in range(args.num_cameras):
        cam_dir = scene_dir / f"camera_{cam:04d}"
        cam_dir.mkdir(exist_ok=True)
        H_out = given[cam] if decision == "store_as_is" else np.linalg.inv(given[cam])
        with open(cam_dir / "calibration.json", "w") as f:
            json.dump({"homography matrix": H_out.tolist()}, f, indent=2)
        print(f"Wrote {cam_dir / 'calibration.json'}")


if __name__ == "__main__":
    main()
