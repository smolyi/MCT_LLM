"""
Callable graph-query functions for the LLM query interface (query_interface.py).
Kept separate and independently testable so tool correctness is verified
BEFORE wiring up an LLM to call them -- per this project's "verify before
building" convention, an LLM tool-use bug is much harder to diagnose than
a plain function bug.

Each function takes/returns plain JSON-serializable types (the LLM sees
these as tool schemas and their outputs as tool results).
"""
import pickle
from collections import defaultdict

import numpy as np
from sentence_transformers import SentenceTransformer

FPS = 30.0
OTHER_TRAJECTORY_STRIDE = 3  # sample every 3rd frame (~10fps) of the candidate entity's side to bound cost
# A sighting built from fewer detections than this is a track fragment, not a confirmed sustained
# track -- a near-zero-distance "match" against one is much more likely the tracker double-detecting
# the SAME physical person under two ids than a real close pass between two different people.
MIN_RELIABLE_DETECTIONS = 10
# Two distinct human bodies cannot occupy the same ground point -- a match this close is almost
# certainly the same physical box counted under two different track ids, regardless of track length.
DUPLICATE_DETECTION_DISTANCE_M = 0.15


def format_mmss(seconds: float) -> str:
    """M:SS -- computed here rather than left to the LLM since this session already found it making
    arithmetic mistakes (e.g. misjudging whether 0.06m exceeded a 0.69m threshold) on far simpler math."""
    total = round(seconds)
    return f"{total // 60}:{total % 60:02d}"


class GraphTools:
    def __init__(self, graph_path: str, embedding_model: str = "all-MiniLM-L6-v2"):
        with open(graph_path, "rb") as f:
            self.G = pickle.load(f)
        self.entities = {
            d["global_id"]: d for n, d in self.G.nodes(data=True) if d["type"] == "entity"
        }
        self.embedder = SentenceTransformer(embedding_model)
        gids = [gid for gid, d in self.entities.items() if "appearance_caption" in d]
        captions = [self.entities[gid]["appearance_caption"] for gid in gids]
        self._caption_gids = gids
        self._caption_embeddings = self.embedder.encode(captions, normalize_embeddings=True) if captions else np.zeros((0, 384))

    def _sightings_of(self, gid: int) -> list:
        return [d for n, d in self.G.nodes(data=True) if d["type"] == "sighting" and d["global_id"] == gid]

    def _frame_pos_by_camera(self, gid: int) -> dict:
        """camera -> {frame: (time_sec, world_x, world_y, sighting_num_detections)} across all of
        gid's sightings. num_detections is carried per-frame so proximity checks can tell a match
        against a solid, multi-second track apart from one against a 2-frame fragment."""
        out = defaultdict(dict)
        for d in self._sightings_of(gid):
            for frame, t, wx, wy in d.get("trajectory", []):
                out[d["camera"]][frame] = (t, wx, wy, d["num_detections"])
        return out

    def search_by_appearance(self, description: str, min_similarity: float = 0.7, top_k: int = 20) -> list:
        """Find ALL entities whose appearance caption semantically matches a text description at or
        above min_similarity (not just the single best match) -- given known ~45x per-person track
        fragmentation in this pipeline, a description usually corresponds to several different
        fragment entities, not one. top_k is just a safety cap on how many to return."""
        if len(self._caption_gids) == 0:
            return []
        query_emb = self.embedder.encode([description], normalize_embeddings=True)[0]
        sims = self._caption_embeddings @ query_emb
        idx = np.argsort(-sims)
        idx = [i for i in idx if sims[i] >= min_similarity][:top_k]
        return [
            {
                "global_id": int(self._caption_gids[i]),
                "appearance_caption": self.entities[self._caption_gids[i]]["appearance_caption"],
                "similarity": round(float(sims[i]), 3),
                "caption_agreement": self.entities[self._caption_gids[i]].get("caption_agreement"),
            }
            for i in idx
        ]

    def get_entity_timeline(self, global_id: int) -> dict:
        """Full sighting history for one entity: which cameras, when, where (world coords)."""
        sightings = sorted(self._sightings_of(global_id), key=lambda d: d["start_frame"])
        if not sightings:
            return {"error": f"no entity with global_id={global_id}"}
        return {
            "global_id": global_id,
            "appearance_caption": self.entities.get(global_id, {}).get("appearance_caption"),
            "caption_agreement": self.entities.get(global_id, {}).get("caption_agreement"),
            "sightings": [
                {
                    "camera": d["camera"],
                    "start_time": format_mmss(d["start_time_sec"]),
                    "end_time": format_mmss(d["end_time_sec"]),
                    "start_time_sec": round(d["start_time_sec"], 1),
                    "end_time_sec": round(d["end_time_sec"], 1),
                    "world_x": round(d["mean_world_x"], 2),
                    "world_y": round(d["mean_world_y"], 2),
                    "num_detections": d["num_detections"],
                }
                for d in sightings
            ],
        }

    def find_entities_in_camera(self, camera_id: str, start_time_sec: float = None, end_time_sec: float = None) -> list:
        """Entities seen in a given camera, optionally restricted to a time window (seconds)."""
        camera_id = str(int(camera_id))  # normalize "0535" or "535" -> "535" (matches graph's stored ids)
        out = []
        for n, d in self.G.nodes(data=True):
            if d["type"] != "sighting" or d["camera"] != camera_id:
                continue
            if start_time_sec is not None and d["end_time_sec"] < start_time_sec:
                continue
            if end_time_sec is not None and d["start_time_sec"] > end_time_sec:
                continue
            out.append({
                "global_id": d["global_id"],
                "appearance_caption": self.entities.get(d["global_id"], {}).get("appearance_caption"),
                "start_time": format_mmss(d["start_time_sec"]),
                "end_time": format_mmss(d["end_time_sec"]),
                "start_time_sec": round(d["start_time_sec"], 1),
                "end_time_sec": round(d["end_time_sec"], 1),
            })
        return out

    def check_entities_cooccur(self, global_id_a: int, global_id_b: int, max_gap_sec: float = 5.0) -> dict:
        """Whether two entities were ever seen in the SAME camera with overlapping (or near-overlapping) time windows."""
        sightings_a = self._sightings_of(global_id_a)
        sightings_b = self._sightings_of(global_id_b)
        matches = []
        for sa in sightings_a:
            for sb in sightings_b:
                if sa["camera"] != sb["camera"]:
                    continue
                gap = max(sa["start_time_sec"] - sb["end_time_sec"], sb["start_time_sec"] - sa["end_time_sec"])
                if gap <= max_gap_sec:
                    matches.append({
                        "camera": sa["camera"],
                        "entity_a_window": f"{format_mmss(sa['start_time_sec'])}-{format_mmss(sa['end_time_sec'])}",
                        "entity_b_window": f"{format_mmss(sb['start_time_sec'])}-{format_mmss(sb['end_time_sec'])}",
                    })
        return {"cooccurred": len(matches) > 0, "matches": matches}

    def find_nearby_entities(self, global_id: int, max_distance_m: float = 2.0,
                              max_gap_sec: float = 1.0, top_k: int = 1000) -> list:
        """Entities whose real-world trajectory actually came within max_distance_m of
        global_id's, in the same camera at (near-)matching timestamps -- a true spatial
        proximity check using per-frame world coordinates, not just same-camera/same-time
        co-occurrence (see check_entities_cooccur, which ignores distance entirely).
        top_k defaults effectively unbounded (well above this scene's ~750-entity scale) -- a low
        default here was the exact same class of bug as rank_entities_by_interaction_count's earlier
        top_k=10 truncation: an entity with 40 real confirmed matches would silently show only 10,
        with no signal to the caller that anything was cut off.
        Per-interactor interaction counts (e.g. "for each of these, how many interactions do THEY
        have") are deliberately NOT precomputed here -- this is a general tool, and the LLM is
        expected to call it again per interactor itself when a query needs that next level of
        detail, the same way it composes any other multi-step query."""
        target_by_cam = self._frame_pos_by_camera(global_id)
        if not any(target_by_cam.values()):
            return {"error": f"no entity with global_id={global_id}"}

        max_gap_frames = int(round(max_gap_sec * FPS))
        candidate_gids = {
            d["global_id"] for n, d in self.G.nodes(data=True)
            if d["type"] == "sighting" and d["global_id"] != global_id and d["camera"] in target_by_cam
        }

        results = []
        for other_gid in candidate_gids:
            best = None
            for cam, frame_pos in self._frame_pos_by_camera(other_gid).items():
                target_frames = target_by_cam.get(cam)
                if not target_frames:
                    continue
                for frame_o, (t_o, wx_o, wy_o, ndet_o) in list(frame_pos.items())[::OTHER_TRAJECTORY_STRIDE]:
                    for df in range(-max_gap_frames, max_gap_frames + 1):
                        t_pos = target_frames.get(frame_o + df)
                        if t_pos is None:
                            continue
                        t_t, wx_t, wy_t, ndet_t = t_pos
                        dist = ((wx_o - wx_t) ** 2 + (wy_o - wy_t) ** 2) ** 0.5
                        if dist <= max_distance_m and (best is None or dist < best["distance_m"]):
                            mid_t = (t_o + t_t) / 2
                            best = {
                                "camera": cam, "time": format_mmss(mid_t), "time_sec": round(mid_t, 1),
                                "distance_m": round(dist, 2),
                                "target_track_detections": ndet_t, "other_track_detections": ndet_o,
                            }
            if best is not None:
                thin = min(best["target_track_detections"], best["other_track_detections"]) < MIN_RELIABLE_DETECTIONS
                too_close = best["distance_m"] <= DUPLICATE_DETECTION_DISTANCE_M
                result = {
                    "global_id": other_gid,
                    "appearance_caption": self.entities.get(other_gid, {}).get("appearance_caption"),
                    "caption_agreement": self.entities.get(other_gid, {}).get("caption_agreement"),
                    **best,
                }
                if too_close:
                    # NOT a proximity match at all -- two distinct human bodies cannot occupy the same
                    # ground point, so this is almost certainly ONE real person, tracked under two ids
                    # and captioned twice (often inconsistently). Describing it as "X was near Y" is
                    # itself the wrong claim; the right claim is "X and Y are probably the same person."
                    result["match_type"] = "likely_same_person"
                    result["note"] = (
                        f"distance ({best['distance_m']}m) is at or below {DUPLICATE_DETECTION_DISTANCE_M}m -- two "
                        "distinct people cannot occupy the same ground point. This is almost certainly ONE real "
                        "person seen under two different track ids (often with two different, sometimes "
                        "contradictory captions), NOT two different people who happened to meet."
                    )
                elif thin:
                    result["match_type"] = "uncertain_proximity"
                    result["note"] = (
                        "one of the two tracks at this match is a short fragment "
                        f"(<{MIN_RELIABLE_DETECTIONS} detections) -- this could be a real brief encounter, "
                        "but a short track makes the position estimate unreliable, so treat this as "
                        "uncertain rather than a confirmed encounter"
                    )
                else:
                    result["match_type"] = "confirmed_proximity"
                results.append(result)
        results.sort(key=lambda r: r["distance_m"])
        # Split by match_type BEFORE capping to top_k -- capping the merged list first was a real bug:
        # near-zero-distance duplicate-detection artifacts are always closest by construction, so they
        # filled every slot and silently pushed genuine, real-distance matches out of the result
        # entirely. Each bucket gets its own top_k budget instead.
        return {
            # Echoed back so callers/formatters can label the target side of a match without having
            # to separately remember which global_id was searched -- find_nearby_entities_by_description
            # already carries this per-match as searched_entity_*; a direct find_nearby_entities call
            # has no other way to know it (this was a real bug: "likely_same_person" pairs rendered as
            # "global_id ? (?)" for the target side when this tool was called directly).
            "target_global_id": global_id,
            "target_caption": self.entities.get(global_id, {}).get("appearance_caption"),
            "confirmed_proximity_matches": [r for r in results if r["match_type"] == "confirmed_proximity"][:top_k],
            "uncertain_proximity_matches": [r for r in results if r["match_type"] == "uncertain_proximity"][:top_k],
            "likely_same_person_matches": [r for r in results if r["match_type"] == "likely_same_person"][:top_k],
        }

    def count_nearby_entities(self, global_id: int, max_distance_m: float = 2.0, max_gap_sec: float = 1.0) -> dict:
        """Just the confirmed-interaction COUNT for one entity, not the full match detail -- a
        general 'how many' primitive, useful any time a query needs a number rather than a list.
        Reuses find_nearby_entities' own computation so the count is always consistent with what that
        tool would report; exists as a separate tool because asking for full detail on every one of an
        entity's own interactors (e.g. 'for each interactor, how many interactions do THEY have') was
        observed, via debug trace, to make the model fetch entire proximity-match dumps for every
        interactor instead of just their counts -- correct but unreadably verbose. A lightweight
        count-only call keeps that kind of compound query's answer readable."""
        result = self.find_nearby_entities(global_id, max_distance_m, max_gap_sec)
        if "error" in result:
            return result
        return {
            "global_id": global_id,
            "appearance_caption": result.get("target_caption"),
            "caption_agreement": self.entities.get(global_id, {}).get("caption_agreement"),
            "confirmed_interaction_count": len(result["confirmed_proximity_matches"]),
        }

    def find_nearby_entities_by_description(self, description: str, min_similarity: float = 0.7,
                                             max_distance_m: float = 2.0, max_gap_sec: float = 1.0,
                                             top_k: int = 1000) -> dict:
        """search_by_appearance + find_nearby_entities, but for EVERY matching candidate, not just
        the best one -- combined into one call because asking the LLM to repeat find_nearby_entities
        for each search_by_appearance candidate across separate tool-call turns proved unreliable in
        practice (it kept stopping after checking only the top-similarity candidate). This loop is
        purely mechanical, so it belongs in code, not in a multi-turn prompt."""
        candidates = self.search_by_appearance(description, min_similarity=min_similarity)
        if not candidates:
            return {"candidates_checked": [], "confirmed_proximity_matches": [],
                     "uncertain_proximity_matches": [], "likely_same_person_matches": []}

        best_by_other_gid = {}
        for cand in candidates:
            nearby = self.find_nearby_entities(cand["global_id"], max_distance_m=max_distance_m, max_gap_sec=max_gap_sec)
            all_matches = (nearby["confirmed_proximity_matches"] + nearby["uncertain_proximity_matches"]
                           + nearby["likely_same_person_matches"])
            for m in all_matches:
                # Careful: this match has TWO different people's captions in it, don't let them get
                # swapped -- "global_id"/"appearance_caption" (from find_nearby_entities) describe the
                # OTHER, nearby person; the searched_entity_* fields below describe the person who
                # actually matched the text description.
                entry = {**m, "searched_entity_global_id": cand["global_id"],
                          "searched_entity_caption": cand["appearance_caption"],
                          "searched_entity_caption_agreement": cand.get("caption_agreement")}
                other_gid = m["global_id"]
                if other_gid not in best_by_other_gid or m["distance_m"] < best_by_other_gid[other_gid]["distance_m"]:
                    best_by_other_gid[other_gid] = entry

        matches = sorted(best_by_other_gid.values(), key=lambda r: r["distance_m"])
        # Split by match_type BEFORE capping to top_k -- same bug/fix as find_nearby_entities: capping
        # the merged list first let near-zero artifacts crowd out genuine real-distance matches.
        return {
            "candidates_checked": [c["global_id"] for c in candidates],
            "confirmed_proximity_matches": [m for m in matches if m["match_type"] == "confirmed_proximity"][:top_k],
            "uncertain_proximity_matches": [m for m in matches if m["match_type"] == "uncertain_proximity"][:top_k],
            "likely_same_person_matches": [m for m in matches if m["match_type"] == "likely_same_person"][:top_k],
        }

    def list_multi_camera_entities(self) -> list:
        """Entities confirmed (by cross-camera ReID matching) to have appeared in more than one camera."""
        by_entity_cams = {}
        for n, d in self.G.nodes(data=True):
            if d["type"] == "sighting":
                by_entity_cams.setdefault(d["global_id"], set()).add(d["camera"])
        return [
            {"global_id": gid, "cameras": sorted(cams), "appearance_caption": self.entities.get(gid, {}).get("appearance_caption")}
            for gid, cams in by_entity_cams.items() if len(cams) > 1
        ]

    def rank_entities_by_interaction_count(self, max_distance_m: float = 2.0, max_gap_sec: float = 1.0,
                                            top_k: int = 10) -> list:
        """Rank ALL entities by how many CONFIRMED proximity matches they have (genuine encounters with
        distinct people -- excludes uncertain/short-fragment matches and likely_same_person artifacts).
        This is a well-defined, computable-in-code answer to "who has the most interactions" -- added
        because the alternative (the LLM calling find_nearby_entities once per entity across many tool
        turns, then counting/ranking the results in free text) is infeasible at this entity count and
        unreliable at the counting step even if it weren't (see: this session's list_multi_camera_entities
        miscount, 281 actual vs 200 reported)."""
        counts = []
        for gid in self.entities:
            # top_k=10_000 here is NOT the caller's top_k (that's applied to the final ranking below) --
            # find_nearby_entities' own top_k defaults to 10 and would silently truncate any entity's
            # match list before we ever count it, making every busy entity tie at exactly 10 (a real bug
            # caught by inspection: every result came back as precisely 10 until this was fixed).
            result = self.find_nearby_entities(gid, max_distance_m=max_distance_m, max_gap_sec=max_gap_sec, top_k=10_000)
            n = len(result.get("confirmed_proximity_matches", []))
            if n > 0:
                counts.append({
                    "global_id": gid,
                    "appearance_caption": self.entities[gid].get("appearance_caption"),
                    "caption_agreement": self.entities[gid].get("caption_agreement"),
                    "confirmed_interaction_count": n,
                })
        counts.sort(key=lambda r: r["confirmed_interaction_count"], reverse=True)
        return counts[:top_k]
