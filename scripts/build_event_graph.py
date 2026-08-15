"""
Build a NetworkX event graph from the merged cross-camera prediction file
(pred_full.txt from build_pred_file.py + cross_camera_reid.py's global
id map): one node per real-world entity (global id), one node per
contiguous "sighting" (a run of frames where that entity was seen in one
camera, allowing small gaps for missed detections), PRECEDES edges
between temporally-ordered sightings of the same entity, and edges
linking sightings to camera nodes.

Schema:
  entity:<gid>          -- a real-world person (as best resolved by tracking+ReID)
  sighting:<gid>:<cam>:<seg_idx> -- one continuous appearance in one camera
  camera:<cam>

  entity --HAS_SIGHTING--> sighting
  sighting --IN_CAMERA--> camera
  sighting --PRECEDES--> sighting   (same entity, time-ordered, across cameras too)

30fps is assumed (confirmed via cv2 on the source videos) to convert
frame numbers to seconds for human-readable time ranges.

Usage:
  python scripts/build_event_graph.py --scene_dir data/scene_061 --pred data/scene_061/pred_full.txt \
      --out data/scene_061/event_graph.gpickle
"""
import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import networkx as nx

FPS = 30.0
MAX_GAP_FRAMES = 30  # allow up to 1s of missed detections within one continuous sighting


def group_into_sightings(frames_with_pos: list) -> list:
    """frames_with_pos: sorted [(frame, world_x, world_y, bbox...), ...] for one (entity, camera).
    Returns segments: consecutive runs allowing gaps <= MAX_GAP_FRAMES."""
    segments = []
    current = [frames_with_pos[0]]
    for rec in frames_with_pos[1:]:
        if rec[0] - current[-1][0] <= MAX_GAP_FRAMES:
            current.append(rec)
        else:
            segments.append(current)
            current = [rec]
    segments.append(current)
    return segments


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    # (camera, global_id) -> sorted list of (frame, world_x, world_y, bbox_left, bbox_top, bbox_w, bbox_h)
    by_entity_camera = defaultdict(list)
    with open(args.pred) as f:
        for line in f:
            cam, gid, frame, l, t, w, h, wx, wy = line.split()
            by_entity_camera[(int(gid), cam)].append(
                (int(frame), float(wx), float(wy), float(l), float(t), float(w), float(h))
            )
    for k in by_entity_camera:
        by_entity_camera[k].sort(key=lambda r: r[0])

    G = nx.DiGraph()

    entities = sorted({gid for gid, _ in by_entity_camera})
    for gid in entities:
        G.add_node(f"entity:{gid}", type="entity", global_id=gid)

    cameras = sorted({cam for _, cam in by_entity_camera})
    for cam in cameras:
        G.add_node(f"camera:{cam}", type="camera", camera_id=cam)

    sightings_by_entity = defaultdict(list)  # gid -> [(start_frame, sighting_node_id), ...]
    for (gid, cam), frames_with_pos in by_entity_camera.items():
        segments = group_into_sightings(frames_with_pos)
        for seg_idx, seg in enumerate(segments):
            start_frame, end_frame = seg[0][0], seg[-1][0]
            mean_wx = sum(r[1] for r in seg) / len(seg)
            mean_wy = sum(r[2] for r in seg) / len(seg)
            # Per-frame (frame, time_sec, world_x, world_y) -- needed for real spatial-proximity
            # checks between two entities' trajectories, not just the mean position.
            trajectory = [(r[0], round(r[0] / FPS, 3), r[1], r[2]) for r in seg]
            node_id = f"sighting:{gid}:{cam}:{seg_idx}"
            G.add_node(
                node_id, type="sighting", global_id=gid, camera=cam,
                start_frame=start_frame, end_frame=end_frame,
                start_time_sec=start_frame / FPS, end_time_sec=end_frame / FPS,
                mean_world_x=mean_wx, mean_world_y=mean_wy,
                trajectory=trajectory,
                num_detections=len(seg),
            )
            G.add_edge(f"entity:{gid}", node_id, relation="HAS_SIGHTING")
            G.add_edge(node_id, f"camera:{cam}", relation="IN_CAMERA")
            sightings_by_entity[gid].append((start_frame, node_id))

    n_precedes = 0
    for gid, sightings in sightings_by_entity.items():
        sightings.sort(key=lambda x: x[0])
        for (_, a), (_, b) in zip(sightings, sightings[1:]):
            G.add_edge(a, b, relation="PRECEDES")
            n_precedes += 1

    print(f"Graph: {len(entities)} entities, {len(cameras)} cameras, "
          f"{sum(1 for n, d in G.nodes(data=True) if d['type']=='sighting')} sightings, "
          f"{n_precedes} PRECEDES edges, {G.number_of_edges()} total edges")

    with open(args.out, "wb") as f:
        pickle.dump(G, f)
    print(f"Saved graph to {args.out}")


if __name__ == "__main__":
    main()
