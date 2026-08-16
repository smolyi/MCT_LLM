"""
Identity resolution: load every camera's track_embeddings.npz (from
extract_track_embeddings.py), compute cosine similarity between tracks,
and merge matched tracks into one global identity.

Two passes, WITHIN-camera first, then cross-camera -- added after CLAUDE.md
documented within-camera track fragmentation (not cross-camera matching) as
the dominant cause of entity over-counting on every scene so far (e.g. 20
true identities vs. 809 predicted on scene_061). This script originally only
ever matched ACROSS cameras; nothing stitched a single camera's own
ByteTrack fragments (ID swaps from occlusion/crowd density) back together.

Matching is done as one-to-one MAXIMUM-WEIGHT BIPARTITE MATCHING (Hungarian
algorithm) per camera PAIR (or, for the within-camera pass, per camera
against itself), thresholded, rather than naive union-find over every
pairwise similarity above a cutoff. That distinction matters a lot in
practice: union-find over all thresholded pairs is equivalent to
single-linkage clustering, which is notorious for "chaining" -- if A~B and
B~C are both just barely above threshold, A and C get merged transitively
even if directly dissimilar. On this dataset that chaining alone collapsed
~1200 tracks into a single global cluster even at threshold=0.6-0.8, before
this fix. Hungarian matching per pair enforces at most one match per track
per pair, which is far more conservative and standard for MTMC. Global
identities are still formed via union-find, but only over this much sparser,
higher-precision set of one-to-one matched pairs -- extending the SAME
conservative mechanism to the within-camera pass rather than reverting
toward global thresholded matching for it.

Layers a spatio-temporal plausibility factor on top of appearance
similarity (see git history for the appearance-only version). Two tracks
get matched on appearance_similarity * plausibility, where plausibility in
[0,1] comes from the walking speed required to cover the gap between their
(median world position, time window):
  - Two DIFFERENT-camera tracks that overlap in time but sit farther apart
    than OVERLAP_DISTANCE_TOLERANCE_M get plausibility=0 -- a hard
    exclusion, since one physical person cannot be in two places at the
    same instant regardless of how similar their embeddings look. Two
    cameras CAN legitimately see the same person from different angles at
    the same time, though, so overlap alone isn't disqualifying cross-camera.
  - Two SAME-camera tracks that overlap in time AT ALL get plausibility=0,
    unconditionally -- a stricter rule than cross-camera, and deliberately
    so: two bounding boxes visible simultaneously in the SAME camera are, by
    construction, two different people, regardless of how close together
    they are (unlike cross-camera, there's no "different angle" excuse for
    simultaneous same-camera visibility of one physical person). Confirmed
    by direct code reading before implementing -- reusing the cross-camera
    rule unchanged here would have let two genuinely different, simultaneously
    -visible people within OVERLAP_DISTANCE_TOLERANCE_M of each other merge.
  - Two non-overlapping tracks (same- or different-camera) get full credit
    up to COMFORTABLE_SPEED_MPS (an average walking pace) and linearly
    decreasing credit up to MAX_SPEED_MPS, beyond which plausibility is 0.
Uses each track's whole-track MEDIAN position (from extract_track_
embeddings.py), not its exact hand-off-point position, since that's what's
already computed -- a coarser proxy, but the tracks in this scene are
mostly short, so the median is usually close to any single point on them.

Within-camera groups are consolidated (mean-renormalized embedding, min/max
frame range, mean world position) BEFORE the cross-camera pass runs, so
Hungarian's one-match-per-track constraint can't strand a track's true
within-camera partner in a separate pairing -- matching happens between
GROUPS, not raw tracks, once within-camera stitching has run. The same
union-find structure carries both passes: cross-camera matches union the
groups' RAW representative keys, which already transitively fuses every
member each group absorbed in the within-camera pass.

Evaluate with evaluate_tracking.py on the merged scene to see how this
changes DetA/AssA vs. either baseline (wrong merges hurt AssA the same way
missed merges do, just differently).

Usage:
  python scripts/cross_camera_reid.py --scene_dir data/scene_061 --cameras 0535 0536 0537 0538 \
      --threshold 0.6 --same_camera_threshold 0.75 --out data/scene_061/global_id_map.json
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
OVERLAP_DISTANCE_TOLERANCE_M = 3.0  # simultaneous CROSS-CAMERA tracks farther apart than this
# can't be one person. Same-camera overlap is a hard veto regardless of distance -- see
# plausibility_matrix's same_camera argument.


SAME_CAMERA_OVERLAP_GRACE_FRAMES = 2  # real bug found via direct visual inspection: a boundary-
# touching same-camera track pair (track A's last frame == track B's first frame -- a classic
# ID-switch handoff, e.g. POM camera_0000 tracks 419/488, confirmed to be the literal SAME
# detection of the SAME person by pulling both crops at that exact shared frame) was being hard-
# vetoed as "simultaneous" by the same_camera overlap rule, because `first_b <= last_a` is True
# at an exact boundary touch (0 frames of real shared visibility). That's the single most common
# and important case the within-camera stitching pass exists to fix -- the previous version was
# blocking it. A GENUINE two-different-simultaneously-visible-people case spans many frames of
# real shared visibility (both people being tracked stably at once), not a single touching frame,
# so a small grace window here doesn't risk conflating the two.


def plausibility_matrix(first_a, last_a, pos_a, first_b, last_b, pos_b, same_camera: bool) -> np.ndarray:
    """(n_a, n_b) matrix in [0, 1]: spatio-temporal plausibility that track i and track j are the
    same physical person, independent of appearance. 0 = physically impossible; 1 = fully plausible
    (or too little separation to rule out). same_camera=True applies a strictly harder overlap
    veto (real simultaneous overlap, beyond SAME_CAMERA_OVERLAP_GRACE_FRAMES, is disqualifying) --
    see module docstring for why this can't be the same rule used cross-camera."""
    dist = np.linalg.norm(pos_a[:, None, :] - pos_b[None, :, :], axis=2)  # (n_a, n_b) meters

    overlap = (first_a[:, None] <= last_b[None, :]) & (first_b[None, :] <= last_a[:, None])
    gap_frames = np.maximum(first_b[None, :] - last_a[:, None], first_a[:, None] - last_b[None, :])
    gap_sec = np.clip(gap_frames, 1, None) / FPS  # floor at 1 frame to avoid div-by-zero

    required_speed = dist / gap_sec
    plaus = np.clip(1 - (required_speed - COMFORTABLE_SPEED_MPS) / (MAX_SPEED_MPS - COMFORTABLE_SPEED_MPS), 0, 1)
    if same_camera:
        overlap_start = np.maximum(first_a[:, None], first_b[None, :])
        overlap_end = np.minimum(last_a[:, None], last_b[None, :])
        real_overlap = (overlap_end - overlap_start) > SAME_CAMERA_OVERLAP_GRACE_FRAMES
        plaus[real_overlap] = 0.0  # real simultaneous visibility disqualifies -- two boxes
        # genuinely visible at once in one camera are always two different people, no distance
        # exception -- but a boundary touch/tiny graze is an ID-switch handoff, not this.
    else:
        # NOTE, found while writing this session's regression tests (not fixed -- pre-existing
        # behavior, unrelated to what this session's within-camera work touched, and changing it
        # would affect already-reported baselines): this explicit veto is almost redundant for
        # cross-camera pairs. gap_sec is floored at 1 frame for any OVERLAPPING pair, so the
        # GENERIC required_speed formula above already drives plaus to ~0 for any overlapping pair
        # beyond roughly 0.1m apart -- well inside OVERLAP_DISTANCE_TOLERANCE_M (3.0m). In practice,
        # an overlapping cross-camera pair only gets real credit when genuinely near-zero distance
        # apart; this line mainly matters for pairs beyond the tolerance that the generic formula
        # hadn't already zeroed for some other reason.
        plaus[overlap & (dist > OVERLAP_DISTANCE_TOLERANCE_M)] = 0.0
    return plaus


HANDOFF_GAP_FRAMES = 8  # same-camera "clean handoff" window -- FULL confidence up to this gap,
# matching the real confirmed cases (gap<=8 frames) this was calibrated against exactly, not a
# looser bound with a linear decay already eating into it.
HANDOFF_GAP_MARGIN_FRAMES = 6  # confidence decays from 1.0 to 0.0 over this ADDITIONAL margin
# beyond HANDOFF_GAP_FRAMES (i.e. reaches 0 at gap=14) -- a real bug found and fixed here: the
# first version decayed linearly from frame 0, so a confirmed real case at gap=6 (well inside the
# calibrated range) only got a 0.4 gap-factor, nowhere near enough combined with the equally
# undercooked distance factor. Confidence should be FULL across the calibrated range, decaying
# only beyond it.
HANDOFF_PIXEL_DIST = 36  # same-camera "clean handoff" pixel-space distance -- FULL confidence up
# to this distance, matching the real confirmed cases (up to 36px) this was calibrated against
# exactly. PIXEL space, not world coordinates -- found empirically that homography-projected world
# distance stayed large (100s of units) even for genuine clean handoffs (ground-plane projection
# accuracy degrades away from the camera), while pixel distance stayed small and consistent across
# every confirmed real case.
HANDOFF_PIXEL_MARGIN = 20  # confidence decays from 1.0 to 0.0 over this ADDITIONAL margin beyond
# HANDOFF_PIXEL_DIST (i.e. reaches 0 at 56px) -- same fix as HANDOFF_GAP_MARGIN_FRAMES.
HANDOFF_OVERRIDE_SIM = 0.85  # effective similarity granted to a same-camera pair with a clean
# handoff, regardless of a possibly-noisy appearance embedding -- confirmed via direct visual
# inspection that several real same-person handoffs had appearance similarity as low as 0.59
# (likely motion blur/tiny-crop noise in the specific sampled frame, not real appearance
# difference), which no reasonable appearance-only threshold could accept without also accepting
# many genuine false positives elsewhere.

GROUP_SIM_TOP_K = 3  # see match_pairs' group_members docstring -- mean of the top-K pairwise
# member similarities between two groups, not the single max (which collapsed 3 distinct real
# groups into one 182-member group via a multiple-comparisons effect: SOME spuriously high pair
# among thousands of cross-pairs between two large groups is found almost every time). Requiring
# several independent pairs to agree is much more robust while still fixing the original diagnosed
# bug (a single representative under-representing a large group).


def match_pairs(keys_a: list, keys_b: list, embeddings: dict, world_pos: dict,
                 first_frame: dict, last_frame: dict, threshold: float, same_camera: bool,
                 caption_embeddings: dict = None, w_appearance: float = 0.8, w_caption: float = 0.2,
                 first_pixel_pos: dict = None, last_pixel_pos: dict = None,
                 group_members: dict = None) -> list:
    """One-to-one Hungarian matching between keys_a and keys_b, thresholded on
    (w_appearance * appearance_similarity + w_caption * caption_similarity) * spatio-temporal
    plausibility. caption_embeddings is optional -- per user direction to match identities by
    appearance, wearing (caption), AND position, not appearance alone; combined ADDITIVELY rather
    than as another hard multiplicative gate, since captions are noisier/more generic than
    appearance embeddings and a bad caption shouldn't be able to veto an otherwise-strong appearance
    match. When caption_embeddings is None (not yet extracted for this scene) or a key is missing
    from it, that pair's score is appearance-only (w_appearance effectively renormalized to 1.0 for
    that pair) rather than penalized for missing data. Returns [(key_a, key_b), ...] pairs to union.
    When same_camera=True, keys_a and keys_b are the SAME list (a camera matched against itself) --
    the diagonal (a track matched to itself) is excluded explicitly rather than relying on
    plausibility_matrix's same-camera overlap veto to zero it incidentally, since a track always
    overlaps itself and would otherwise get plausibility=0 there anyway, but explicit is safer
    against future changes to that rule.

    When same_camera=True and first_pixel_pos/last_pixel_pos are given, a "clean handoff" signal
    (tiny time gap AND tiny pixel-space displacement between track A's last detection and track B's
    first) can OVERRIDE a weak appearance score -- see HANDOFF_* constants. This never LOWERS a
    pair's score, only rescues cases where the spatiotemporal evidence alone is already strong
    enough to be confident despite a noisy embedding.

    group_members (optional, {representative_key: [all member keys]}) fixes a real bug found via
    direct evidence, not assumed: each key in keys_a/keys_b is normally ONE representative embedding
    standing in for a whole (possibly large, already-merged) group, but a SINGLE representative
    embedding badly under-represents a large group -- confirmed concretely on POM camera_0000: many
    small groups had similarity as low as 0.6-0.8 to a 60+-member group's chosen representative, but
    0.88-0.95 to SOME OTHER member of that same group. When group_members is given, sim[i,j] is the
    MAX appearance similarity across every member pair between group i and group j, not just their
    two representatives -- still fed into the SAME one-to-one-per-round Hungarian matching (not a
    reversion to naive any-pair-above-threshold clustering, which is what caused the original
    catastrophic chaining collapse), just with a similarity signal that doesn't collapse a
    many-member group down to one arbitrary point."""
    if not keys_a or not keys_b:
        return []
    if group_members is not None:
        # Real bug found and fixed here, not shipped blind: raw MAX similarity across all member
        # pairs was tried first, but between two LARGE groups (60-83 members each) the number of
        # cross-pairs explodes (thousands), and a classic multiple-comparisons effect means SOME
        # spuriously high pairwise similarity is found almost every time even between genuinely
        # different people -- confirmed concretely: this collapsed 3 previously-distinct groups
        # (including a visually-confirmed brown-coat person and a visually-confirmed white-top
        # person) into one 182-member group. Fixed by using the MEAN of the top-K pairwise
        # similarities instead of the single max -- still resistant to one representative under-
        # representing a group (the original diagnosed bug: several genuinely-good cross-pairs
        # exist for a real match, not just one), but requires multiple independent pairs to agree,
        # not just one lucky/unlucky outlier.
        emb_by_group_a = [np.stack([embeddings[m] for m in group_members.get(k, [k])]) for k in keys_a]
        emb_by_group_b = [np.stack([embeddings[m] for m in group_members.get(k, [k])]) for k in keys_b]

        def _top_k_mean(pair_sims: np.ndarray, k: int = GROUP_SIM_TOP_K) -> float:
            flat = pair_sims.flatten()
            k = min(k, flat.size)
            return float(np.mean(np.sort(flat)[-k:]))

        sim = np.array([[_top_k_mean(ea @ eb.T) for eb in emb_by_group_b] for ea in emb_by_group_a])
    else:
        emb_a = np.stack([embeddings[k] for k in keys_a])
        emb_b = np.stack([embeddings[k] for k in keys_b])
        sim = emb_a @ emb_b.T

    if caption_embeddings:
        has_cap_a = np.array([k in caption_embeddings for k in keys_a])
        has_cap_b = np.array([k in caption_embeddings for k in keys_b])
        both_have_caps = has_cap_a[:, None] & has_cap_b[None, :]
        cap_a = np.stack([caption_embeddings.get(k, np.zeros_like(next(iter(caption_embeddings.values())))) for k in keys_a])
        cap_b = np.stack([caption_embeddings.get(k, np.zeros_like(next(iter(caption_embeddings.values())))) for k in keys_b])
        cap_sim = cap_a @ cap_b.T
        combined_sim = np.where(both_have_caps, w_appearance * sim + w_caption * cap_sim, sim)
    else:
        combined_sim = sim

    handoff_score = None
    if same_camera and first_pixel_pos is not None and last_pixel_pos is not None:
        # Missing entries (a key from a camera whose track_embeddings.npz predates the pixel-
        # position fields) default to an infinite placeholder -- handoff_confidence naturally goes
        # to 0 for those pairs (a safe no-op fallback to appearance-only), never a crash.
        _inf_px = np.array([np.inf, np.inf])
        has_px_a = np.array([k in last_pixel_pos for k in keys_a])
        has_px_b = np.array([k in first_pixel_pos for k in keys_b])
        both_have_px = has_px_a[:, None] & has_px_b[None, :]
        gap_frames = np.maximum(np.array([first_frame[k] for k in keys_b])[None, :] - np.array([last_frame[k] for k in keys_a])[:, None],
                                 np.array([first_frame[k] for k in keys_a])[:, None] - np.array([last_frame[k] for k in keys_b])[None, :])
        last_px_a = np.stack([last_pixel_pos.get(k, _inf_px) for k in keys_a])
        first_px_b = np.stack([first_pixel_pos.get(k, _inf_px) for k in keys_b])
        px_dist = np.linalg.norm(last_px_a[:, None, :] - first_px_b[None, :, :], axis=2)
        # FULL confidence (1.0) anywhere inside the calibrated range, decaying to 0 only over the
        # margin beyond it -- NOT a linear decay from zero (see HANDOFF_GAP_MARGIN_FRAMES's
        # docstring for the real bug this replaced: a confirmed real case sitting near the edge of
        # the calibrated range was getting almost no credit under the original formula).
        gap_factor = np.clip(1 - np.maximum(0, gap_frames - HANDOFF_GAP_FRAMES) / HANDOFF_GAP_MARGIN_FRAMES, 0, 1)
        dist_factor = np.clip(1 - np.maximum(0, px_dist - HANDOFF_PIXEL_DIST) / HANDOFF_PIXEL_MARGIN, 0, 1)
        handoff_confidence = gap_factor * dist_factor
        handoff_confidence = np.where(both_have_px, handoff_confidence, 0.0)
        handoff_score = handoff_confidence * HANDOFF_OVERRIDE_SIM

    plaus = plausibility_matrix(
        np.array([first_frame[k] for k in keys_a]), np.array([last_frame[k] for k in keys_a]),
        np.stack([world_pos[k] for k in keys_a]),
        np.array([first_frame[k] for k in keys_b]), np.array([last_frame[k] for k in keys_b]),
        np.stack([world_pos[k] for k in keys_b]),
        same_camera=same_camera,
    )
    combined = combined_sim * plaus
    if handoff_score is not None:
        # Applied AFTER the plausibility multiplication, not before -- real bug found and fixed
        # here: the median-world-position-based plausibility formula is subject to the SAME
        # projection-accuracy problem documented for the handoff distance itself (a long, wandering
        # track's median position can sit hundreds of world-units from its actual endpoint),
        # so a genuinely clean pixel-space handoff could still get multiplied by a near-zero
        # plausibility computed from that same unreliable median position. A confirmed clean
        # handoff (tight time+space match) already IS the physical-plausibility evidence -- it
        # doesn't need to also pass a separate, less reliable plausibility check for the same claim.
        combined = np.maximum(combined, handoff_score)
    if same_camera and keys_a is keys_b:
        # Real (previously latent) bug found while writing this session's tests: this used to fire
        # on same_camera=True alone, regardless of whether keys_a and keys_b were actually the same
        # list -- harmless in production (always called as match_pairs(reps, reps, ...), so keys_a
        # IS keys_b there), but WRONG in general: with two different single-element lists, "the
        # diagonal" is position (0,0), which does NOT mean "a key matched to itself" unless the two
        # lists are actually the same object. Checking identity explicitly instead of trusting the
        # same_camera flag alone.
        np.fill_diagonal(combined, -1.0)  # a track can never match itself

    row_idx, col_idx = linear_sum_assignment(-combined)  # maximize combined score
    pairs = []
    for r, c in zip(row_idx, col_idx):
        if combined[r, c] >= threshold:
            pairs.append((keys_a[r], keys_b[c]))
    return pairs


def merge_would_violate_same_camera_overlap(all_keys: list, uf: "UnionFind", key_a, key_b,
                                             first_frame: dict, last_frame: dict) -> bool:
    """True if unioning key_a's and key_b's groups would put two overlapping-in-time raw tracks
    from the SAME camera into one global entity -- physically impossible (one person can't be in
    two places in one camera at once), and NOT already prevented by the same-camera hard veto in
    plausibility_matrix, which only guards a single Hungarian call's own direct matches. Two
    DIFFERENT camera-pair matches during the cross-camera pass (e.g. group A matches group C via
    camera-pair (0,2), then group B separately matches group C via camera-pair (1,2)) can still
    transitively fuse A and B through their shared root even if A and B were never compared to each
    other directly -- if A and B both contain same-camera-overlapping raw tracks, that transitive
    fusion would silently reintroduce exactly the impossibility the hard veto exists to prevent.
    Checked empirically to not yet occur on any of this project's 3 scenes at current thresholds,
    but not something to rely on by chance -- enforced explicitly here instead."""
    root_a, root_b = uf.find(key_a), uf.find(key_b)
    members_a = [k for k in all_keys if uf.find(k) == root_a]
    members_b = [k for k in all_keys if uf.find(k) == root_b]
    for ma in members_a:
        for mb in members_b:
            if ma[0] != mb[0]:  # different cameras -- cross-camera overlap is fine
                continue
            if first_frame[ma] <= last_frame[mb] and first_frame[mb] <= last_frame[ma]:
                return True
    return False


def consolidate_group_representatives(keys: list, uf: "UnionFind", embeddings: dict,
                                       first_frame: dict, last_frame: dict, world_pos: dict) -> tuple:
    """Collapses `keys` (already-partially-unioned via `uf`) into one representative key per group:
    the LARGEST member's (longest frame span) embedding and frame range, used AS-IS -- not pooled or
    widened, per the two real bugs documented in main() (mean-pooling dilutes the appearance signal;
    min/max frame-range widening spuriously triggers the cross-camera overlap veto). Only world
    position, which doesn't have either failure mode, is averaged across all members. Returns
    (reps, group_members): one representative raw key per group (unioning representatives elsewhere
    still fuses the full group, since uf already ties every member to the same root), plus a
    {representative_key: [all member keys]} map -- needed by match_pairs' group_members argument to
    compute max-pooled similarity instead of collapsing a large group to one point (see its
    docstring for the real bug this fixes)."""
    groups = {}  # root -> [member keys]
    for key in keys:
        groups.setdefault(uf.find(key), []).append(key)
    reps = []
    group_members = {}
    for root, members in groups.items():
        rep = max(members, key=lambda m: last_frame[m] - first_frame[m])
        reps.append(rep)
        group_members[rep] = members
        if len(members) > 1:
            world_pos[rep] = np.mean([world_pos[m] for m in members], axis=0)
    return reps, group_members


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
    parser.add_argument("--threshold", type=float, default=0.6, help="Cross-camera match threshold.")
    parser.add_argument("--same_camera_threshold", type=float, default=0.75,
                         help="Within-camera match threshold for ROUND 1 -- kept separate from "
                              "--threshold since same-camera fragment pairs (typically a short gap "
                              "right after an ID swap) have a different plausibility distribution "
                              "than cross-camera pairs; needs its own sweep per scene, not a reuse "
                              "of the cross-camera default.")
    parser.add_argument("--same_camera_threshold_increment", type=float, default=0.03,
                         help="Added to --same_camera_threshold on each subsequent within-camera "
                              "stitching round (capped at 0.95). Iterative stitching (see main() "
                              "docstring) is needed to resolve multi-way fragmentation chains a "
                              "single Hungarian pass can't -- but a consolidated group's "
                              "representative gets noisier (less like any single real crop) the "
                              "more merging it's already absorbed, so later rounds should demand "
                              "higher similarity, not the same threshold as round 1. Calibrated "
                              "against a real regression: round-1-threshold applied uniformly at "
                              "every round produced visibly wrong merges by round ~5-9 on POM "
                              "camera_0000 (confirmed via direct visual inspection -- 2 of 6 "
                              "spot-checked members of the largest resulting group were different "
                              "real people). Set to 0 to reproduce that flat-threshold behavior.")
    parser.add_argument("--skip_same_camera", action="store_true",
                         help="Disable the within-camera stitching pass (old behavior, for A/B comparison).")
    parser.add_argument("--use_caption_similarity", action="store_true",
                         help="Enable the caption-similarity matching signal (requires "
                              "track_captions.npz per camera, from extract_track_captions.py). "
                              "OFF by default and must be passed explicitly -- real bug found "
                              "and fixed here: this used to auto-activate whenever "
                              "track_captions.npz files happened to exist on disk, regardless of "
                              "the documented 'disabled by default' decision (see CLAUDE.md's "
                              "Phase 3 writeup -- the caption signal measurably increases "
                              "over-merging risk on populations with generic clothing). A later "
                              "re-run on POM after Phase 3's files were already on disk silently "
                              "re-enabled it and produced a different (more-merged) result than "
                              "the intended appearance-only default, caught by comparing entity "
                              "counts against the documented canonical result rather than assumed "
                              "safe. File presence is no longer sufficient on its own.")
    parser.add_argument("--w_appearance", type=float, default=0.8,
                         help="Weight on appearance-embedding similarity when --use_caption_similarity "
                              "is set (see match_pairs docstring).")
    parser.add_argument("--w_caption", type=float, default=0.2,
                         help="Weight on caption-text similarity, additive with --w_appearance. "
                              "Starts conservative per the approved plan -- only raise if spot-checks "
                              "show it isn't causing over-merging.")
    parser.add_argument("--use_group_pooled_similarity", action="store_true",
                         help="Use match_pairs' group_members max/top-K-pooled similarity instead of "
                              "single-representative-to-representative for within-camera matching. "
                              "OFF by default -- real regression found and reverted, not shipped on "
                              "the strength of its motivating diagnosis alone: correctly fixes a "
                              "confirmed under-merging bug (a single representative badly "
                              "under-represents a large, already-merged group) for SMALL-vs-LARGE "
                              "group comparisons, but between two LARGE groups the number of cross-"
                              "pairs explodes and OSNet's own confirmed weak discrimination on this "
                              "dataset (documented same-camera-different-track similarity up to 0.96 "
                              "between confirmed DIFFERENT people) means a multiple-comparisons "
                              "effect finds a spuriously high pair almost every time -- confirmed "
                              "concretely: this merged 3 visually-distinct real people (including a "
                              "confirmed brown-coat person and a confirmed white-top person) into one "
                              "182-member group on POM camera_0000, even using a top-3-mean instead "
                              "of raw max. Left available for a future, more discriminative ReID "
                              "model, but not trustworthy with the current one.")
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    scene_dir = Path(args.scene_dir)

    all_keys, all_embeddings = [], []
    keys_by_camera = {}
    world_pos, first_frame, last_frame = {}, {}, {}
    first_pixel_pos, last_pixel_pos = {}, {}  # only populated for cameras whose track_embeddings.npz
    # has been re-run with the pixel-position fields (see extract_track_embeddings.py) -- older
    # .npz files silently fall back to appearance+world-plausibility-only same-camera matching.
    caption_embeddings = {}  # (cam, track_id) -> embedding, only populated if --use_caption_similarity
    for cam in args.cameras:
        data = scene_dir / f"camera_{cam}" / "track_embeddings.npz"
        data = np.load(data)
        has_pixel_pos = "first_pixel_positions" in data.files
        keys_by_camera[cam] = []
        for i, (tid, emb, pos, ff, lf) in enumerate(zip(
            data["track_ids"], data["embeddings"], data["world_positions"],
            data["first_frames"], data["last_frames"],
        )):
            key = (cam, int(tid))
            all_keys.append(key)
            all_embeddings.append(emb)
            keys_by_camera[cam].append(key)
            world_pos[key] = pos
            first_frame[key] = int(ff)
            last_frame[key] = int(lf)
            if has_pixel_pos:
                first_pixel_pos[key] = data["first_pixel_positions"][i]
                last_pixel_pos[key] = data["last_pixel_positions"][i]
        if not has_pixel_pos:
            print(f"  (no pixel-position data for camera_{cam} -- re-run extract_track_embeddings.py "
                  f"to enable the same-camera clean-handoff signal; falling back to appearance-only "
                  f"for this camera's within-camera matching)")

        if not args.use_caption_similarity:
            continue
        captions_path = scene_dir / f"camera_{cam}" / "track_captions.npz"
        if captions_path.exists():
            cap_data = np.load(captions_path)
            for tid, emb in zip(cap_data["track_ids"], cap_data["caption_embeddings"]):
                caption_embeddings[(cam, int(tid))] = emb
        else:
            print(f"  (no track_captions.npz for camera_{cam} -- that camera's pairs will be "
                  f"appearance-only; run extract_track_captions.py to enable the caption signal)")
    embeddings = {k: e for k, e in zip(all_keys, all_embeddings)}

    print(f"{len(all_keys)} total tracks across {len(args.cameras)} cameras")

    uf = UnionFind(all_keys)

    # PASS 1: within-camera stitching, runs FIRST -- collapses each camera's own ByteTrack
    # fragments before the cross-camera pass sees them (see module docstring for why this
    # ordering matters for Hungarian's one-match-per-track constraint).
    #
    # ITERATIVE, not one-shot -- real bug found via direct visual inspection (not shipped as a
    # single Hungarian pass blindly): a single all-vs-all Hungarian round enforces at most ONE
    # match per track, so a person fragmented into 3+ pieces in one round can only pair up at most
    # floor(n/2) of them, even if every pairwise similarity is well above threshold. Confirmed
    # concretely on POM camera_0000: tracks 419 and 488 are the SAME detection at the SAME frame
    # (a track ID split with zero real gap) with cosine similarity 0.765 (above the 0.75
    # threshold) -- but Hungarian's single-round GLOBAL optimum matched 419->749 and 488->928
    # instead (its own better-scoring pairs elsewhere), leaving the 419/488 link never made even
    # though it was individually well above threshold. Re-running the same conservative mechanism
    # (Hungarian one-to-one + threshold, never global transitive clustering) repeatedly, on the
    # shrinking set of GROUP representatives, lets exactly this kind of multi-way fragmentation
    # chain resolve over several rounds instead of being capped at one pairing per track.
    n_same_camera_merges = 0
    if not args.skip_same_camera:
        for cam in args.cameras:
            round_num = 0
            while True:
                round_num += 1
                round_threshold = min(
                    args.same_camera_threshold + (round_num - 1) * args.same_camera_threshold_increment, 0.95)
                reps, group_members = consolidate_group_representatives(keys_by_camera[cam], uf, embeddings,
                                                                          first_frame, last_frame, world_pos)
                pairs = match_pairs(reps, reps, embeddings, world_pos, first_frame, last_frame,
                                     round_threshold, same_camera=True,
                                     caption_embeddings=caption_embeddings,
                                     w_appearance=args.w_appearance, w_caption=args.w_caption,
                                     first_pixel_pos=first_pixel_pos, last_pixel_pos=last_pixel_pos,
                                     group_members=group_members if args.use_group_pooled_similarity else None)
                if not pairs:
                    break
                for key_a, key_b in pairs:
                    uf.union(key_a, key_b)
                    n_same_camera_merges += 1
            if round_num > 1:
                print(f"  camera_{cam}: within-camera stitching converged after {round_num} rounds "
                      f"(final round threshold={round_threshold:.2f})")
        print(f"Within-camera stitching: {n_same_camera_merges} pairwise merges "
              f"(threshold={args.same_camera_threshold})")

    # Consolidate each camera's post-stitching groups into one representative per group for the
    # cross-camera pass. Two real bugs found and fixed here, not shipped blind, both surfaced by
    # comparing against a --skip_same_camera A/B run on the identical (crop-quality-filtered)
    # embeddings: cross-camera spanning matches collapsed from 121 (no within-camera stitching) to
    # 13 (with it, first version).
    # Bug 1 (embeddings): the first version MEAN-pooled every group member's embedding. Averaging
    # multiple crops of genuinely the same person can land in a "blended" region of embedding space
    # that's LESS similar to a correct cross-camera match than any single strong fragment would be
    # -- and averaging a WRONGLY-merged group (different real people) makes this worse. Fixed by
    # using the LARGEST member's (longest frame span, a proxy for most detections / most reliable
    # observation) embedding as-is, no averaging. This alone barely moved the number (13 -> 12) --
    # NOT the dominant cause, though still a real improvement to keep.
    # Bug 2 (frame range -- the actual dominant cause): also consolidated first/last frame to
    # min/max across ALL group members, which artificially WIDENS a group's temporal footprint --
    # e.g. two fragments at frames 1000-1200 and 5000-5200 become one group spanning [1000, 5200].
    # plausibility_matrix's cross-camera "overlap" branch then spuriously fires against unrelated
    # candidate tracks anywhere in that wide window; and even when NOT hard-vetoed, its generic
    # required_speed=dist/gap_sec formula floors gap_sec at 1 frame for ANY overlapping pair,
    # collapsing plausibility to ~0 for all but near-zero distances -- so widening the window
    # silently converts what should be legitimate "sequential, walkable gap" cross-camera candidates
    # into spurious "overlapping, implausible" ones before Hungarian ever sees them. Fixed by using
    # the SAME representative member's own (tight, unwidened) first/last frame instead of the
    # group's min/max envelope -- consistent with also using that member's own embedding. Reuses
    # the same consolidate_group_representatives() helper the iterative within-camera loop above
    # uses each round -- this is just its final call, after within-camera stitching has converged.
    group_keys_by_camera = {
        cam: consolidate_group_representatives(keys_by_camera[cam], uf, embeddings, first_frame, last_frame, world_pos)[0]
        for cam in args.cameras
    }  # [0] = reps only -- cross-camera matching doesn't (yet) use the group_members max-pooling
    # fix; within-camera groups here are typically smaller than the within-camera pass's own
    # largest groups mid-iteration, so the single-representative issue is less acute, but worth
    # revisiting if cross-camera precision issues turn up later.

    n_groups = sum(len(v) for v in group_keys_by_camera.values())
    print(f"{n_groups} within-camera groups (from {len(all_keys)} raw tracks) entering cross-camera matching")

    # PASS 2: cross-camera matching, on the consolidated groups from pass 1.
    n_merges = 0
    n_blocked_by_invariant = 0
    for cam_a, cam_b in combinations(args.cameras, 2):
        keys_a, keys_b = group_keys_by_camera[cam_a], group_keys_by_camera[cam_b]
        pairs = match_pairs(keys_a, keys_b, embeddings, world_pos, first_frame, last_frame,
                             args.threshold, same_camera=False,
                             caption_embeddings=caption_embeddings,
                             w_appearance=args.w_appearance, w_caption=args.w_caption)
        for key_a, key_b in pairs:
            # Explicit invariant check, not just relying on chance -- see
            # merge_would_violate_same_camera_overlap's docstring for why this can't just be left
            # to the same-camera hard veto in plausibility_matrix (that only guards ONE Hungarian
            # call's direct matches, not transitive fusion across several).
            if merge_would_violate_same_camera_overlap(all_keys, uf, key_a, key_b, first_frame, last_frame):
                n_blocked_by_invariant += 1
                continue
            uf.union(key_a, key_b)
            n_merges += 1
    print(f"Cross-camera matching: {n_merges} pairwise merges (threshold={args.threshold}, "
          f"{n_blocked_by_invariant} candidate merges blocked by the same-camera-overlap invariant)")

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
    print(f"{n_clusters} final global identities from {len(all_keys)} raw tracks "
          f"({n_cross_camera_clusters} of them span >1 camera)")

    with open(args.out, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"Saved mapping to {args.out}")


if __name__ == "__main__":
    main()
