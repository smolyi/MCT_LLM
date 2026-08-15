"""
Cross-camera identity resolution: load every camera's track_embeddings.npz
(from extract_track_embeddings.py), compute cosine similarity between
tracks from DIFFERENT cameras, and merge matched tracks into one global
identity.

Matching is done as one-to-one MAXIMUM-WEIGHT BIPARTITE MATCHING (Hungarian
algorithm) per camera PAIR, thresholded, rather than naive union-find over
every pairwise similarity above a cutoff. That distinction matters a lot in
practice: union-find over all thresholded pairs is equivalent to
single-linkage clustering, which is notorious for "chaining" -- if A~B and
B~C are both just barely above threshold, A and C get merged transitively
even if directly dissimilar. On this dataset that chaining alone collapsed
~1200 tracks into a single global cluster even at threshold=0.6-0.8, before
this fix. Hungarian matching per camera pair enforces at most one match per
track per pair, which is far more conservative and standard for MTMC.
Global identities are still formed via union-find, but only over this much
sparser, higher-precision set of one-to-one matched pairs.

Second pass: layers a spatio-temporal plausibility factor on top of the
appearance-only baseline (see git history for that version). Two tracks in
different cameras get matched on appearance_similarity * plausibility,
where plausibility in [0,1] comes from the walking speed required to cover
the gap between their (median world position, time window):
  - Two tracks that overlap in time but sit farther apart than
    OVERLAP_DISTANCE_TOLERANCE_M get plausibility=0 -- a hard exclusion,
    since one physical person cannot be in two places at the same instant
    regardless of how similar their embeddings look.
  - Two non-overlapping tracks get full credit up to COMFORTABLE_SPEED_MPS
    (an average walking pace) and linearly decreasing credit up to
    MAX_SPEED_MPS, beyond which plausibility is also 0.
Uses each track's whole-track MEDIAN position (from extract_track_
embeddings.py), not its exact hand-off-point position, since that's what's
already computed -- a coarser proxy, but the tracks in this scene are
mostly short, so the median is usually close to any single point on them.
Evaluate with evaluate_tracking.py on the merged 4-camera scene to see how
this changes DetA/AssA vs the appearance-only baseline (wrong merges hurt
AssA the same way missed merges do, just differently).

Usage:
  python scripts/cross_camera_reid.py --scene_dir data/scene_061 --cameras 0535 0536 0537 0538 \
      --threshold 0.6 --out data/scene_061/global_id_map.json
"""
import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

FPS = 30.0
COMFORTABLE_SPEED_MPS = 1.5  # average walking pace -- full plausibility credit up to here
MAX_SPEED_MPS = 3.0  # brisk walk/slow jog upper bound -- plausibility hits 0 at/above this
OVERLAP_DISTANCE_TOLERANCE_M = 3.0  # simultaneous tracks farther apart than this can't be one person


def plausibility_matrix(first_a, last_a, pos_a, first_b, last_b, pos_b) -> np.ndarray:
    """(n_a, n_b) matrix in [0, 1]: spatio-temporal plausibility that track i (camera A) and
    track j (camera B) are the same physical person, independent of appearance. 0 = physically
    impossible; 1 = fully plausible (or too little separation to rule out)."""
    dist = np.linalg.norm(pos_a[:, None, :] - pos_b[None, :, :], axis=2)  # (n_a, n_b) meters

    overlap = (first_a[:, None] <= last_b[None, :]) & (first_b[None, :] <= last_a[:, None])
    gap_frames = np.maximum(first_b[None, :] - last_a[:, None], first_a[:, None] - last_b[None, :])
    gap_sec = np.clip(gap_frames, 1, None) / FPS  # floor at 1 frame to avoid div-by-zero

    required_speed = dist / gap_sec
    plaus = np.clip(1 - (required_speed - COMFORTABLE_SPEED_MPS) / (MAX_SPEED_MPS - COMFORTABLE_SPEED_MPS), 0, 1)
    plaus[overlap & (dist > OVERLAP_DISTANCE_TOLERANCE_M)] = 0.0
    return plaus


class UnionFind:
    def __init__(self, items):
        self.parent = {x: x for x in items}

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_dir", type=str, required=True)
    parser.add_argument("--cameras", type=str, nargs="+", required=True)
    parser.add_argument("--threshold", type=float, default=0.6)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    scene_dir = Path(args.scene_dir)

    all_keys, all_embeddings = [], []
    keys_by_camera = {}
    world_pos, first_frame, last_frame = {}, {}, {}
    for cam in args.cameras:
        data = scene_dir / f"camera_{cam}" / "track_embeddings.npz"
        data = np.load(data)
        keys_by_camera[cam] = []
        for tid, emb, pos, ff, lf in zip(
            data["track_ids"], data["embeddings"], data["world_positions"],
            data["first_frames"], data["last_frames"],
        ):
            key = (cam, int(tid))
            all_keys.append(key)
            all_embeddings.append(emb)
            keys_by_camera[cam].append(key)
            world_pos[key] = pos
            first_frame[key] = int(ff)
            last_frame[key] = int(lf)
    embeddings = {k: e for k, e in zip(all_keys, all_embeddings)}

    print(f"{len(all_keys)} total tracks across {len(args.cameras)} cameras")

    uf = UnionFind(all_keys)
    n_merges = 0
    for cam_a, cam_b in combinations(args.cameras, 2):
        keys_a, keys_b = keys_by_camera[cam_a], keys_by_camera[cam_b]
        if not keys_a or not keys_b:
            continue
        emb_a = np.stack([embeddings[k] for k in keys_a])
        emb_b = np.stack([embeddings[k] for k in keys_b])
        sim = emb_a @ emb_b.T

        plaus = plausibility_matrix(
            np.array([first_frame[k] for k in keys_a]), np.array([last_frame[k] for k in keys_a]),
            np.stack([world_pos[k] for k in keys_a]),
            np.array([first_frame[k] for k in keys_b]), np.array([last_frame[k] for k in keys_b]),
            np.stack([world_pos[k] for k in keys_b]),
        )
        combined = sim * plaus

        row_idx, col_idx = linear_sum_assignment(-combined)  # maximize combined score
        for r, c in zip(row_idx, col_idx):
            if combined[r, c] >= args.threshold:
                uf.union(keys_a[r], keys_b[c])
                n_merges += 1

    # Assign compact global ids to each connected component.
    root_to_global_id = {}
    mapping = {}  # "cam,track_id" -> global_id
    next_id = 0
    for key in all_keys:
        root = uf.find(key)
        if root not in root_to_global_id:
            root_to_global_id[root] = next_id
            next_id += 1
        mapping[f"{key[0]},{key[1]}"] = root_to_global_id[root]

    n_clusters = len(root_to_global_id)
    n_cross_camera_clusters = sum(
        1 for gid in set(mapping.values())
        if len({k.split(",")[0] for k, v in mapping.items() if v == gid}) > 1
    )
    print(f"threshold={args.threshold}: {n_merges} pairwise merges -> {n_clusters} global identities "
          f"({n_cross_camera_clusters} of them span >1 camera)")

    with open(args.out, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"Saved mapping to {args.out}")


if __name__ == "__main__":
    main()
