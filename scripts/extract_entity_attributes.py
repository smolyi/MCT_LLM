"""
Attribute extraction: for each entity in the event graph, sample several
crops spread across ALL its sightings (not just one), caption each with
BLIP, and store both a consensus caption and an agreement score as node
attributes.

Single-crop captioning (see git history) was a documented source of real
errors this project found directly by inspecting actual video frames: e.g.
one entity's sole crop got captioned "a woman in a red dress is playing
tennis" when the crop actually showed a person in a red TOP + jeans
standing on a plain floor -- there is no tennis anywhere in this dataset,
and BLIP free-associated it from a blue grid-tiled floor texture. A single
bad/ambiguous/partial crop had no way to be caught or corrected before this.

Multiple crops let disagreement itself become a signal, following the same
"uncertainty is information" pattern already used for tracking (low_confidence
proximity matches, spatio-temporal ReID plausibility): if several crops of
the same entity produce very different captions, that's useful information
(this entity's description is unreliable), not noise to average away.

Consensus caption = the sampled caption whose sentence-embedding is closest
to the mean of all sampled captions' embeddings (the "medoid") -- more
robust than an arbitrary single crop, and avoids trying to average
embeddings and decode the result, which BLIP can't do (not invertible).
caption_agreement = mean pairwise cosine similarity across the sampled
captions' embeddings, in [0, 1] -- low agreement flags a caption as
contested even though the query interface can't independently verify it.

Uses Salesforce/blip-image-captioning-base rather than ../LMM_dive's
TinyLMM: that project's own POPE evaluation found TinyLMM answers "yes"
to essentially every yes/no question regardless of ground truth (0%
"no"-accuracy) -- not reliable enough for descriptions we intend to
actually query against. BLIP is a real, evaluated captioning model.

Usage:
  python scripts/extract_entity_attributes.py --scene_dir data/scene_061 --graph data/scene_061/event_graph.gpickle \
      --out data/scene_061/event_graph_with_attrs.gpickle
"""
import argparse
import pickle
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from sentence_transformers import SentenceTransformer
from transformers import BlipForConditionalGeneration, BlipProcessor

CROPS_PER_ENTITY = 3


def pick_sample_frames(frames: list, k: int) -> list:
    """k evenly-spaced (camera, frame) pairs from a list sorted by frame number, dedup'd."""
    if len(frames) <= k:
        return frames
    idxs = np.linspace(0, len(frames) - 1, k).round().astype(int)
    seen, out = set(), []
    for i in idxs:
        if frames[i] not in seen:
            seen.add(frames[i])
            out.append(frames[i])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_dir", type=str, required=True)
    parser.add_argument("--graph", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--crops_per_entity", type=int, default=CROPS_PER_ENTITY)
    args = parser.parse_args()

    scene_dir = Path(args.scene_dir)
    with open(args.graph, "rb") as f:
        G = pickle.load(f)

    # Gather every sighting per entity, across ALL cameras -- not just the single largest one.
    sightings_by_entity = defaultdict(list)  # gid -> [(node_id, data), ...]
    for n, d in G.nodes(data=True):
        if d["type"] == "sighting":
            sightings_by_entity[d["global_id"]].append((n, d))

    print(f"Sampling up to {args.crops_per_entity} crops each for {len(sightings_by_entity)} entities...")

    # Read pred_full.txt once, index by (camera, frame) -> list of (gid, l, t, w, h)
    pred_path = scene_dir / "pred_full.txt"
    by_cam_frame = defaultdict(list)
    with open(pred_path) as f:
        for line in f:
            cam, gid, frame, l, t, w, h, wx, wy = line.split()
            by_cam_frame[(cam, int(frame))].append((int(gid), float(l), float(t), float(w), float(h)))

    # For each entity, pick sample (camera, frame) points spread across ALL its sightings.
    sample_targets_by_camera = defaultdict(list)  # camera -> [(gid, frame), ...]
    for gid, sightings in sightings_by_entity.items():
        all_frames = []  # (frame, camera) -- sort key first so spacing is chronological-ish
        for n, d in sightings:
            for frame, t, wx, wy in d.get("trajectory", []):
                all_frames.append((frame, d["camera"]))
        all_frames.sort(key=lambda x: x[0])
        for frame, cam in pick_sample_frames(all_frames, args.crops_per_entity):
            sample_targets_by_camera[cam].append((gid, frame))

    # Extract crops, grouped by camera so each video is opened/seeked once.
    crops_by_gid = defaultdict(list)  # gid -> [np.ndarray, ...]
    for cam, targets in sample_targets_by_camera.items():
        cam_dir = f"camera_{int(cam):04d}"
        cap = cv2.VideoCapture(str(scene_dir / cam_dir / "video.mp4"))
        for gid, frame in targets:
            bbox = None
            for offset in range(0, 16):
                for fr in (frame + offset, frame - offset):
                    for cand_gid, l, t, w, h in by_cam_frame.get((cam, fr), []):
                        if cand_gid == gid:
                            bbox = (fr, l, t, w, h)
                            break
                    if bbox:
                        break
                if bbox:
                    break
            if bbox is None:
                continue
            fr, l, t, w, h = bbox
            cap.set(cv2.CAP_PROP_POS_FRAMES, fr)
            ok, img = cap.read()
            if not ok:
                continue
            h_img, w_img = img.shape[:2]
            x1, y1 = max(0, int(l)), max(0, int(t))
            x2, y2 = min(w_img, int(l + w)), min(h_img, int(t + h))
            if x2 > x1 and y2 > y1:
                crops_by_gid[gid].append(img[y1:y2, x1:x2, ::-1])  # BGR -> RGB
        cap.release()

    n_crops = sum(len(v) for v in crops_by_gid.values())
    print(f"Extracted {n_crops} crops across {len(crops_by_gid)} entities, running BLIP captioning...")

    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to("cuda")
    blip.eval()
    embedder = SentenceTransformer("all-MiniLM-L6-v2")

    n_captioned = 0
    n_low_agreement = 0
    for gid, crops in crops_by_gid.items():
        captions = []
        for crop in crops:
            image = Image.fromarray(crop)
            inputs = processor(image, return_tensors="pt").to("cuda")
            with torch.no_grad():
                out = blip.generate(**inputs, max_new_tokens=30)
            captions.append(processor.decode(out[0], skip_special_tokens=True))

        if len(captions) == 1:
            consensus, agreement = captions[0], 1.0
        else:
            embs = embedder.encode(captions, normalize_embeddings=True)
            mean_emb = embs.mean(axis=0)
            mean_emb /= np.linalg.norm(mean_emb)
            consensus = captions[int(np.argmax(embs @ mean_emb))]
            n = len(captions)
            pairwise_sum = (embs @ embs.T).sum() - n  # exclude self-similarity (diagonal = 1)
            agreement = float(pairwise_sum / (n * (n - 1)))

        G.nodes[f"entity:{gid}"]["appearance_caption"] = consensus
        G.nodes[f"entity:{gid}"]["all_captions"] = captions
        G.nodes[f"entity:{gid}"]["caption_agreement"] = round(agreement, 3)
        n_captioned += 1
        if agreement < 0.6:
            n_low_agreement += 1
        if n_captioned % 100 == 0:
            print(f"  ...{n_captioned}/{len(crops_by_gid)} ({n_low_agreement} low-agreement so far)")

    print(f"Captioned {n_captioned} entities ({n_low_agreement} with caption_agreement < 0.6)")
    with open(args.out, "wb") as f:
        pickle.dump(G, f)
    print(f"Saved graph with attributes to {args.out}")


if __name__ == "__main__":
    main()
