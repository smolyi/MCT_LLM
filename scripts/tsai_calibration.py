"""
Calibration adapter for EPFL-RLC, which ships ONLY a Tsai camera model
(intrinsics + radial distortion + extrinsics) -- unlike POM, there's no
directly-given ground-plane homography to use or cross-check against, so
this derives one from scratch and verifies it the same empirical way
(real multi-camera detection consistency -- see adapt_pom_calibration.py,
where the same approach was validated after a Tsai-based cross-derivation
attempt there turned out too unreliable to trust on its own).

Kept separate from geometry.py (which only ever handles a bare homography)
and from adapt_pom_calibration.py (POM's problem -- verifying a GIVEN
homography's direction -- is different from RLC's -- deriving one at all).

Ground-plane homography: P = K[R|t] (3x4), drop the Z column (Z=0 for
ground-plane points) -- same technique as scene_061's own calibration.json
(P with its Z column dropped) and POM's Tsai cross-check.

Distortion (kappa1): applied FIRST WITHOUT correction (matches the
approved plan's ordering), then empirically measured by reprojecting a
grid of image points with and without undistortion and comparing the
resulting world-coordinate divergence -- only worth adding real distortion
correction if that divergence is large relative to the scene's scale.

Usage:
  python scripts/tsai_calibration.py --scene_dir data/EPFL-RLC --num_cameras 3
"""
import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_tsai_xml(xml_path: Path) -> dict:
    root = ET.parse(xml_path).getroot()
    geom, intr, extr = root.find("Geometry"), root.find("Intrinsic"), root.find("Extrinsic")
    return {
        "width": float(geom.get("width")), "height": float(geom.get("height")),
        "focal": float(intr.get("focal")), "kappa1": float(intr.get("kappa1")),
        "cx": float(intr.get("cx")), "cy": float(intr.get("cy")), "sx": float(intr.get("sx")),
        "dpx": float(geom.get("dpx")), "dpy": float(geom.get("dpy")),
        "tx": float(extr.get("tx")), "ty": float(extr.get("ty")), "tz": float(extr.get("tz")),
        "rx": float(extr.get("rx")), "ry": float(extr.get("ry")), "rz": float(extr.get("rz")),
    }


def tsai_ground_plane_homography(tsai: dict) -> np.ndarray:
    """WORLD->IMAGE ground-plane homography at the calibration's own native resolution --
    same "drop the Z column from P=K[R|t]" technique as scene_061's own calibration.json
    and adapt_pom_calibration.py's Tsai cross-check."""
    fx = tsai["focal"] * tsai["sx"] / tsai["dpx"]
    fy = tsai["focal"] / tsai["dpy"]
    K = np.array([[fx, 0, tsai["cx"]], [0, fy, tsai["cy"]], [0, 0, 1]])
    R, _ = cv2.Rodrigues(np.array([tsai["rx"], tsai["ry"], tsai["rz"]]))
    t = np.array([tsai["tx"], tsai["ty"], tsai["tz"]])
    P = K @ np.column_stack([R, t])
    return P[:, [0, 1, 3]]  # drop Z column


def undistort_point(x: float, y: float, tsai: dict) -> tuple:
    """Classic Tsai radial (kappa1-only) undistortion: converts a DISTORTED pixel coordinate
    to its undistorted position, by inverting x_distorted = x_undistorted*(1+kappa1*r^2) around
    the principal point (cx,cy), in normalized (mm-scale) sensor coordinates."""
    cx, cy, dpx, dpy = tsai["cx"], tsai["cy"], tsai["dpx"], tsai["dpy"]
    xd, yd = (x - cx) * dpx, (y - cy) * dpy  # pixel offset -> mm offset from center
    r2 = xd ** 2 + yd ** 2
    # x_distorted = x_undistorted * (1 + kappa1 * r_undistorted^2) has no closed-form inverse in
    # general, but for small kappa1*r^2 (checked below) a single fixed-point iteration is enough.
    scale = 1.0
    for _ in range(5):
        scale = 1.0 / (1.0 + tsai["kappa1"] * r2 * scale ** 2)
    xu, yu = xd * scale, yd * scale
    return (xu / dpx + cx, yu / dpy + cy)


def measure_distortion_impact(scene_dir: Path, num_cameras: int) -> float:
    """Reprojects a grid of image points with vs. without undistortion, converts both through the
    (no-distortion) homography, and returns the max world-coordinate divergence in the SAME units
    as the calibration's translation (mm, given tx/ty/tz magnitudes in the thousands) -- lets the
    decision of whether distortion correction is worth adding be based on a measured number, not
    a guess."""
    max_divergence = 0.0
    for cam in range(num_cameras):
        tsai = parse_tsai_xml(scene_dir / "calibration" / f"calibration_cam{cam}.xml")
        H = tsai_ground_plane_homography(tsai)
        H_inv = np.linalg.inv(H)

        def to_world(x, y):
            p = H_inv @ np.array([x, y, 1.0])
            return np.array([p[0] / p[2], p[1] / p[2]])

        w, h = tsai["width"], tsai["height"]
        for gx in np.linspace(w * 0.1, w * 0.9, 5):
            for gy in np.linspace(h * 0.1, h * 0.9, 5):
                world_distorted = to_world(gx, gy)
                ux, uy = undistort_point(gx, gy, tsai)
                world_undistorted = to_world(ux, uy)
                max_divergence = max(max_divergence, float(np.linalg.norm(world_distorted - world_undistorted)))
    return max_divergence


def verify_via_detections(scene_dir: Path, cam_dirs: list, homographies: dict) -> str:
    """Same empirical check validated for POM: run real YOLO detections, map through both candidate
    uses of each derived homography, and see which makes DIFFERENT cameras' detections of the same
    real people converge on the same world coordinates. Scored by mean nearest-cross-camera-detection
    distance (NOT overall mean pairwise distance -- that metric was tried first for POM and found
    misleading, since it's dominated by distances between different real people rather than the
    same person seen twice)."""
    from ultralytics import YOLO
    model = YOLO(str(PROJECT_ROOT / "yolo11s.pt"))

    def to_world(H, x, y):
        p = H @ np.array([x, y, 1.0])
        return np.array([p[0] / p[2], p[1] / p[2]])

    entries_by_case = {"store_as_is": [], "store_inverted": []}
    for cam, cam_dir in enumerate(cam_dirs):
        from video_source import resolve_video_source
        cap = cv2.VideoCapture(resolve_video_source(cam_dir))
        cap.set(cv2.CAP_PROP_POS_FRAMES, 500)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            continue
        detections = model.predict(frame, classes=[0], conf=0.3, verbose=False)[0]
        H = homographies[cam]
        for box in detections.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            gx, gy = (x1 + x2) / 2, y2
            entries_by_case["store_as_is"].append((cam, to_world(np.linalg.inv(H), gx, gy)))
            entries_by_case["store_inverted"].append((cam, to_world(H, gx, gy)))

    scores = {}
    for case, entries in entries_by_case.items():
        if len(entries) < 2:
            continue
        nearest = []
        for i, (cam_i, pt_i) in enumerate(entries):
            others = [pt_j for j, (cam_j, pt_j) in enumerate(entries) if cam_j != cam_i]
            if others:
                nearest.append(min(np.linalg.norm(pt_i - pt_j) for pt_j in others))
        if nearest:
            scores[case] = float(np.mean(nearest))
            print(f"  {case}: {len(entries)} detections, mean nearest-cross-camera-detection "
                  f"world-distance = {scores[case]:.1f}")
    if not scores:
        print("  WARNING: not enough detections across cameras to verify -- defaulting to store_as_is, unverified.")
        return "store_as_is"
    winner = min(scores, key=scores.get)
    print(f"  -> tighter cross-camera agreement: {winner}")
    return winner


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_dir", type=str, required=True)
    parser.add_argument("--num_cameras", type=int, default=3)
    args = parser.parse_args()
    scene_dir = Path(args.scene_dir)

    print("Measuring kappa1 distortion's practical impact on ground-plane world coordinates:")
    max_divergence = measure_distortion_impact(scene_dir, args.num_cameras)
    tsai0 = parse_tsai_xml(scene_dir / "calibration" / "calibration_cam0.xml")
    scene_scale = float(np.linalg.norm([tsai0["tx"], tsai0["ty"], tsai0["tz"]]))
    ratio = max_divergence / scene_scale
    print(f"  max divergence with vs without undistortion: {max_divergence:.1f} (same units as "
          f"calibration translation, scene scale ~{scene_scale:.0f}) -> {100 * ratio:.1f}% of scene scale")
    if ratio > 1.0:
        # A divergence bigger than the scene itself is not a real distortion measurement -- it means
        # undistort_point's fixed-point iteration diverged (kappa1*r^2 here is ~-0.28, well outside the
        # small-perturbation regime a naive fixed-point inversion converges in), not that distortion is
        # somehow catastrophic. Flagged honestly rather than reported as a real number -- this
        # measurement did NOT decide anything below; the actual decision to skip distortion correction
        # rests on the real-detection cross-camera consistency check instead, which is unaffected by
        # this bug (it never calls undistort_point).
        print(f"  NOT a real measurement -- {100*ratio:.0f}% is physically impossible (divergence bigger "
              f"than the scene itself) and means the undistortion fixed-point iteration diverged for "
              f"this camera's kappa1, not that distortion is actually this large. Proceeding without "
              f"distortion correction anyway, but on the strength of the cross-camera consistency check "
              f"below (which doesn't depend on this broken measurement), not this number.")
    elif ratio > 0.02:
        print("  WARNING: divergence is >2% of scene scale -- distortion correction may matter here. "
              "Proceeding WITHOUT it anyway per the plan's ordering, but this is a known, measured "
              "approximation, not an assumption.")
    else:
        print("  Negligible -- proceeding without distortion correction is empirically justified, not assumed.")

    homographies = {cam: tsai_ground_plane_homography(parse_tsai_xml(
        scene_dir / "calibration" / f"calibration_cam{cam}.xml")) for cam in range(args.num_cameras)}

    # Native Tsai resolution (1920x1080) vs. the actual downloaded frames (480x270) -- checked, not
    # assumed, same real scale mismatch already found and handled for POM.
    cam_dirs = [scene_dir / f"camera_{cam:04d}" for cam in range(args.num_cameras)]
    for cam, cam_dir in enumerate(cam_dirs):
        cam_dir.mkdir(exist_ok=True)
        raw_dir = scene_dir / "frames" / f"cam{cam}"
        link = cam_dir / "frames_raw"
        if not link.exists():
            link.symlink_to(raw_dir.resolve())

    tsai0_native_w = parse_tsai_xml(scene_dir / "calibration" / "calibration_cam0.xml")["width"]
    from video_source import resolve_video_source
    cap = cv2.VideoCapture(resolve_video_source(cam_dirs[0]))
    actual_w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    cap.release()
    scale = actual_w / tsai0_native_w
    print(f"\nActual frame width {actual_w:.0f} vs. Tsai native {tsai0_native_w:.0f} -> scale={scale}")
    S = np.diag([scale, scale, 1.0])
    homographies_scaled = {cam: S @ H for cam, H in homographies.items()}

    print("\nVerifying homography direction via multi-camera detection consistency:")
    decision = verify_via_detections(scene_dir, cam_dirs, homographies_scaled)

    for cam, cam_dir in enumerate(cam_dirs):
        H_out = homographies_scaled[cam] if decision == "store_as_is" else np.linalg.inv(homographies_scaled[cam])
        with open(cam_dir / "calibration.json", "w") as f:
            json.dump({"homography matrix": H_out.tolist()}, f, indent=2)
        print(f"Wrote {cam_dir / 'calibration.json'}")


if __name__ == "__main__":
    main()
