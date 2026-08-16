"""
Integration tests against the REAL event graph (data/scene_061's
event_graph_with_attrs.gpickle) -- no LLM/model loading required, but these
do load the actual ~750-entity graph and, in a couple of cases, run a real
full-graph scan, so this suite is slower than test_regressions.py (seconds
to a few minutes, not milliseconds). For pure-logic unit tests with no
graph dependency, see test_regressions.py. For tests requiring the LLM
itself, see test_llm_smoke.py.

Requires data/scene_061/event_graph_with_attrs.gpickle to exist (built via
scripts/build_event_graph.py + scripts/extract_entity_attributes.py) --
skipped automatically if it doesn't.

Usage:
  python tests/test_graph_integration.py -v
"""
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

GRAPH_PATH = PROJECT_ROOT / "data" / "scene_061" / "event_graph_with_attrs.gpickle"

from graph_tools import GraphTools, MIN_RELIABLE_DETECTIONS, DUPLICATE_DETECTION_DISTANCE_M


@unittest.skipUnless(GRAPH_PATH.exists(), f"requires the real graph at {GRAPH_PATH}")
class GraphIntegrationTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tools = GraphTools(str(GRAPH_PATH))


class TestFindNearbyEntitiesTopK(GraphIntegrationTestCase):
    """Bug: find_nearby_entities' top_k parameter isn't exposed in the
    LLM-facing tool schema, so the Python DEFAULT is what every real call
    gets -- it used to default to 10, silently truncating any entity with
    more than 10 confirmed matches (a real, observed case: an entity with
    40 real matches showed only 10, with no signal anything was cut off).
    Fixed by raising the default to 1000 (well above this graph's scale)."""

    def test_default_top_k_does_not_truncate_a_busy_entity(self):
        # global_id 7 is a known busy entity in scene_061 (documented in CLAUDE.md/session
        # history as having tens of confirmed proximity matches at the default 2.0m/1.0s).
        result = self.tools.find_nearby_entities(7)
        total_matches = len(result["confirmed_proximity_matches"])
        # The exact count can drift slightly as the pipeline is re-run, but the whole point
        # of the bug is that the OLD default (10) would silently cap it there -- anything
        # meaningfully above 10 proves the truncation regression hasn't returned.
        self.assertGreater(total_matches, 10, "find_nearby_entities may have regressed to top_k=10")

    def test_top_k_default_constant_is_not_the_known_bad_value(self):
        import inspect
        sig = inspect.signature(self.tools.find_nearby_entities)
        self.assertNotEqual(sig.parameters["top_k"].default, 10)


class TestLowQualityCaptionEntities(GraphIntegrationTestCase):
    """Bugs: (1) entity 8 was captioned "a sandwich with a sandwich on it"
    despite 2 of its 3 real sampled crops clearly describing a person --
    fixed via degenerate-caption filtering + majority-vote consensus.
    (2) the tool only reported an entity's first sighting, dropping later
    ones. (3) entities with a very short total track were still reported
    despite too little evidence to trust either way."""

    def test_entity_8_no_longer_a_sandwich(self):
        # Direct visual inspection (screenshot) showed entity 8's frame is a real person --
        # confirms the majority-vote consensus fix picked a human-describing caption instead.
        caption = self.tools.entities[8]["appearance_caption"]
        self.assertNotIn("sandwich", caption.lower())

    def test_every_flagged_entity_has_at_least_one_sighting(self):
        results = self.tools.list_low_quality_caption_entities()
        self.assertGreater(len(results), 0, "expected at least some flagged entities on this dataset")
        for r in results:
            self.assertIn("sightings", r)
            self.assertGreaterEqual(len(r["sightings"]), 1)

    def test_no_flagged_entity_has_a_degenerate_repeated_word_caption(self):
        from extract_entity_attributes import is_degenerate_caption
        results = self.tools.list_low_quality_caption_entities()
        for r in results:
            self.assertFalse(is_degenerate_caption(r["appearance_caption"]),
                              f"global_id {r['global_id']} has a degenerate caption that should"
                              f" have been filtered out of consensus selection: {r['appearance_caption']!r}")

    def test_no_flagged_entity_has_a_too_short_track(self):
        results = self.tools.list_low_quality_caption_entities()
        # Recompute total detections the same way the tool does, directly from sightings, to
        # confirm none of the reported entities fall below the reliability floor.
        for r in results:
            sightings = self.tools._sightings_of(r["global_id"])
            total = sum(s.get("num_detections", 0) for s in sightings)
            self.assertGreaterEqual(total, MIN_RELIABLE_DETECTIONS,
                                     f"global_id {r['global_id']} has too short a track "
                                     f"({total} detections) to have been reported at all")

    def test_every_flagged_caption_has_no_human_word(self):
        results = self.tools.list_low_quality_caption_entities()
        for r in results:
            words = set(r["appearance_caption"].lower().replace(",", " ").replace(".", " ").split())
            self.assertFalse(words & GraphTools.HUMAN_CAPTION_WORDS,
                              f"global_id {r['global_id']} was flagged but its caption "
                              f"{r['appearance_caption']!r} contains a human word")


class TestProximityMatchTypeCategorization(GraphIntegrationTestCase):
    """Bug: the confidence flag missed near-zero-distance false positives
    when both tracks were individually long enough to look "reliable" --
    a 0.01m-apart match with both tracks >= MIN_RELIABLE_DETECTIONS wasn't
    flagged at all. Fixed by adding DUPLICATE_DETECTION_DISTANCE_M as an
    independent physical-implausibility check: two distinct people cannot
    occupy the same ground point, regardless of track length."""

    def test_match_type_categories_are_internally_consistent(self):
        # Spot-check a handful of entities: within confirmed_proximity_matches, no match should
        # actually be at or below the duplicate-detection distance (those belong in
        # likely_same_person_matches instead, by construction).
        checked_any = False
        for gid in list(self.tools.entities.keys())[:30]:
            result = self.tools.find_nearby_entities(gid)
            if "error" in result:
                continue
            for m in result["confirmed_proximity_matches"]:
                checked_any = True
                self.assertGreater(m["distance_m"], DUPLICATE_DETECTION_DISTANCE_M,
                                    f"global_id {gid}'s match {m['global_id']} at {m['distance_m']}m "
                                    f"should have been categorized likely_same_person, not confirmed")
            for m in result["likely_same_person_matches"]:
                checked_any = True
                self.assertLessEqual(m["distance_m"], DUPLICATE_DETECTION_DISTANCE_M)
        self.assertTrue(checked_any, "no matches found across sampled entities -- test may be too narrow")


class TestRankEntitiesByInteractionCount(GraphIntegrationTestCase):
    """Bug: rank_entities_by_interaction_count's internal per-entity calls
    to find_nearby_entities used the OLD default top_k=10, so every busy
    entity's count was silently capped at exactly 10 -- every result came
    back tied at precisely 10 until this was fixed by passing an explicit,
    much larger top_k on the internal call.

    SLOW: this does a real full-graph scan (documented as a couple of
    minutes in CLAUDE.md). Run this test deliberately, not as part of a
    tight edit-test loop."""

    def test_top_entities_are_not_all_tied_at_ten(self):
        ranked = self.tools.rank_entities_by_interaction_count(top_k=5)
        counts = [r["confirmed_interaction_count"] for r in ranked]
        self.assertFalse(all(c == 10 for c in counts),
                          "all top counts are exactly 10 -- the internal top_k truncation bug may have returned")
        # The busiest entities on this dataset are well-documented (session history) to have
        # more than 10 confirmed interactions -- the top one alone should exceed it.
        self.assertGreater(max(counts), 10)


if __name__ == "__main__":
    unittest.main()
