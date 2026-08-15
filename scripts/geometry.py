"""
Image-plane <-> ground-plane geometry, using each camera's calibration.json
homography matrix H (3x3). This is a ground-plane-only mapping (valid for
points that lie on the ground, z=0) -- confirmed by the fact that
`homography matrix` in calibration.json is exactly `camera projection
matrix` with its 3rd column (the Z coefficients) dropped, the standard
construction for a planar homography from a full projection matrix.

Direction, confirmed empirically against ground_truth.txt (not assumed):
H maps WORLD -> IMAGE (image = H @ [x_world, y_world, 1]^T, normalized).
So image -> world requires H's inverse. Also confirmed empirically: the
image point that corresponds to world_x/world_y is the bbox's
bottom-center (ground-contact point), not its center -- projecting a
ground-truth row's bottom-center through H^-1 reproduces its world_x/
world_y to within ~0.1, whereas the bbox center is off by ~1.5.
"""
import json
import numpy as np


def load_homography(calibration_path) -> np.ndarray:
    with open(calibration_path) as f:
        calib = json.load(f)
    return np.array(calib["homography matrix"], dtype=np.float64)


def bbox_ground_point(bbox_left: float, bbox_top: float, bbox_width: float, bbox_height: float) -> tuple:
    """Bottom-center of the bbox -- the point where a standing person/object touches the ground."""
    return (bbox_left + bbox_width / 2, bbox_top + bbox_height)


def image_to_world(x_img: float, y_img: float, homography: np.ndarray) -> tuple:
    h_inv = np.linalg.inv(homography)
    p = h_inv @ np.array([x_img, y_img, 1.0])
    return (p[0] / p[2], p[1] / p[2])
