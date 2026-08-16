"""
Regression tests for specific bugs found and fixed during this project's
development -- each test below is named after (and documents) the real bug
it guards against, per this project's "verify before building, verify after
fixing" convention. These are all FAST, PURE-LOGIC tests: no GPU, no model
loading, no real graph file required -- they test the deterministic Python
code around the LLM/models, which is exactly the class of fix this project
found to be reliable (see CLAUDE.md's "Insights and lessons learned").

For tests that load the real event graph (GraphTools against the actual
data), see test_graph_integration.py. For tests that require the LLM itself
(slow, GPU-dependent), see test_llm_smoke.py.

Usage:
  python -m unittest tests.test_regressions -v
  (or, from the tests/ directory: python -m unittest test_regressions -v)
"""
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np

from color_utils import strip_color_words, dominant_color_name, garment_region, NAMED_COLORS_RGB, MIN_GARMENT_REGION_PIXELS
from crop_quality import is_low_quality_crop, MIN_CROP_AREA_PX
from cross_camera_reid import (
    plausibility_matrix, match_pairs, merge_would_violate_same_camera_overlap, UnionFind,
    consolidate_group_representatives, SAME_CAMERA_OVERLAP_GRACE_FRAMES,
    HANDOFF_GAP_FRAMES, HANDOFF_PIXEL_DIST, HANDOFF_OVERRIDE_SIM,
)
from extract_entity_attributes import is_degenerate_caption, CAPTION_CONFIDENCE_THRESHOLD
from graph_tools import GraphTools
from query_interface import (
    agreement_flag,
    format_count_report,
    format_ranking_report,
    format_low_quality_caption_report,
    format_proximity_report_merged,
    format_interactor_count_nudge,
    format_invented_id_recovery,
    format_deterministic_summary,
    _build_display_reports,
    _strip_fabricated_count_claims,
    _combine_answer,
    _mentions_unexecuted_tool,
    _extract_tool_calls,
    CAPTION_AGREEMENT_THRESHOLD,
)


class TestDegenerateCaptionDetection(unittest.TestCase):
    """Bug: BLIP decoder repetition loops (e.g. "a bald bald bald bald...")
    were eligible to be chosen as the medoid consensus caption over two
    coherent, correct captions -- found via entity 357, whose OTHER two
    sampled captions were "a woman walking down the street with her dog"
    and "a man in a white shirt and tie". Fixed with is_degenerate_caption,
    which excludes repeated-word captions from medoid selection entirely."""

    def test_flags_repeated_word_caption(self):
        self.assertTrue(is_degenerate_caption(
            "a bald bald bald bald bald bald bald bald bald bald"))

    def test_flags_repetition_anywhere_in_caption(self):
        self.assertTrue(is_degenerate_caption("a man man man man is walking"))

    def test_does_not_flag_normal_caption(self):
        self.assertFalse(is_degenerate_caption("a woman walking down the street with her dog"))
        self.assertFalse(is_degenerate_caption("a man in a white shirt and tie"))

    def test_does_not_flag_two_consecutive_repeats(self):
        # Threshold is 3+ in a row -- "sandwich sandwich" (2x) is normal BLIP phrasing
        # ("a sandwich with a sandwich on it"), not decoder degeneration.
        self.assertFalse(is_degenerate_caption("a sandwich with a sandwich on it"))

    def test_empty_caption_does_not_crash(self):
        self.assertFalse(is_degenerate_caption(""))


class TestHumanCaptionWordCoverage(unittest.TestCase):
    """Bug: the human-word heuristic list_low_quality_caption_entities uses
    to decide whether a caption "failed" originally had gaps -- "player"
    was missing (misflagged "a baseball player is throwing a ball" as a
    failed caption), and plural "girls"/"boys" were missing (misflagged
    "two girls are seen in this surveillance image")."""

    def _mentions_human(self, caption: str) -> bool:
        words = set(caption.lower().replace(",", " ").replace(".", " ").split())
        return bool(words & GraphTools.HUMAN_CAPTION_WORDS)

    def test_player_recognized_as_human(self):
        self.assertTrue(self._mentions_human("a baseball player is throwing a ball"))

    def test_plural_girls_recognized_as_human(self):
        self.assertTrue(self._mentions_human("two girls are seen in this surveillance image"))

    def test_plural_boys_recognized_as_human(self):
        self.assertTrue(self._mentions_human("three boys playing in the yard"))

    def test_singular_forms_still_recognized(self):
        for caption in ["a man walking", "a woman running", "a boy on a bike", "a girl with a hat"]:
            self.assertTrue(self._mentions_human(caption), caption)

    def test_genuinely_non_human_caption_not_flagged_as_human(self):
        self.assertFalse(self._mentions_human("a small white bird flying through the air"))


class TestAgreementScoreDisplay(unittest.TestCase):
    """Bug: agreement_flag() only renders text when the score is BELOW
    threshold; a high score renders as an empty string, which was
    misread as "no agreement score exists" rather than "the score is
    fine". format_low_quality_caption_report was fixed to always show the
    numeric score explicitly. Also guards a real NameError this project
    shipped briefly (referencing an undefined `agreement_str` variable)."""

    def test_low_agreement_shows_number_in_flag(self):
        flag = agreement_flag(0.1)
        self.assertIn("0.1", flag)
        self.assertIn("uncertain", flag)

    def test_high_agreement_flag_is_empty_string(self):
        # This is the correct, existing behavior -- the bug was elsewhere (the
        # report formatter) treating this emptiness as "score is missing".
        self.assertEqual(agreement_flag(0.9), "")

    def test_none_agreement_flag_is_empty_string(self):
        self.assertEqual(agreement_flag(None), "")

    def test_report_never_hides_a_high_agreement_score(self):
        result = [{"global_id": 1, "appearance_caption": "a bird flying", "caption_agreement": 0.9,
                   "sightings": [{"camera": "537", "time": "0:01"}], "num_cameras": 1}]
        report = format_low_quality_caption_report(result)
        self.assertIn("0.9", report)  # would previously show NO agreement value at all

    def test_report_handles_none_agreement_without_crashing(self):
        result = [{"global_id": 2, "appearance_caption": "a thing", "caption_agreement": None,
                   "sightings": [{"camera": "537", "time": "0:01"}], "num_cameras": 1}]
        report = format_low_quality_caption_report(result)  # must not raise NameError
        self.assertIn("global_id 2", report)
        self.assertIn("unknown", report)

    def test_report_splits_low_and_higher_confidence_sections(self):
        result = [
            {"global_id": 1, "appearance_caption": "low conf", "caption_agreement": 0.1,
             "sightings": [{"camera": "537", "time": "0:01"}], "num_cameras": 1},
            {"global_id": 2, "appearance_caption": "high conf", "caption_agreement": 0.9,
             "sightings": [{"camera": "537", "time": "0:02"}], "num_cameras": 1},
        ]
        report = format_low_quality_caption_report(result)
        low_idx = report.index("Low-confidence")
        high_idx = report.index("Higher-confidence")
        self.assertLess(low_idx, report.index("global_id 1"))
        self.assertLess(high_idx, report.index("global_id 2"))
        self.assertLess(low_idx, high_idx)


class TestLowQualityCaptionReportSightings(unittest.TestCase):
    """Bug: the tool (then named list_non_human_entities) originally reported
    only an entity's FIRST sighting; later sightings in other cameras were
    silently dropped. Fixed to report every sighting."""

    def test_all_sightings_appear_in_report(self):
        result = [{"global_id": 1, "appearance_caption": "a bird", "caption_agreement": 0.2,
                   "sightings": [{"camera": "535", "time": "1:58"}, {"camera": "536", "time": "2:10"},
                                  {"camera": "537", "time": "3:00"}],
                   "num_cameras": 3}]
        report = format_low_quality_caption_report(result)
        for cam, t in [("535", "1:58"), ("536", "2:10"), ("537", "3:00")]:
            self.assertIn(f"camera {cam} at {t}", report)


class TestFabricationScrubber(unittest.TestCase):
    """Bug: when a large batch of count_nearby_entities calls partially
    failed to parse (garbled <tool_call> tags under greedy decoding), the
    model's NEXT turn sometimes fabricated plausible-looking counts for
    exactly the ids that never got a real result -- sometimes wrapped in a
    fake <tool_response> tag, sometimes as bare prose matching the report
    template closely enough to evade a tag-only check. This is the
    structural backstop: cross-check every count claim against verified
    tool results, not against a text pattern."""

    def test_strips_claim_for_unverified_global_id(self):
        intro = "global_id 337: 11 confirmed interactions of their own -- a man is playing tennis."
        cleaned = _strip_fabricated_count_claims(intro, deterministic_reports=[])
        self.assertNotIn("337: 11 confirmed", cleaned)
        self.assertIn("removed", cleaned)

    def test_preserves_claim_for_verified_global_id(self):
        real_report = "global_id 337: 11 confirmed interactions of their own -- a man is playing tennis."
        intro = "global_id 337: 11 confirmed interactions of their own -- a man is playing tennis."
        cleaned = _strip_fabricated_count_claims(intro, deterministic_reports=[real_report])
        self.assertIn("337: 11 confirmed", cleaned)

    def test_mixed_verified_and_fabricated_claims(self):
        real_report = "global_id 1: 5 confirmed interactions of their own -- a man in a blue shirt."
        intro = ("global_id 1: 5 confirmed interactions of their own -- a man in a blue shirt.\n"
                 "global_id 999: 3 confirmed interactions of their own -- a fabricated entity.")
        cleaned = _strip_fabricated_count_claims(intro, deterministic_reports=[real_report])
        self.assertIn("1: 5 confirmed", cleaned)
        self.assertNotIn("999: 3 confirmed", cleaned)
        self.assertIn("global_id 999", cleaned)  # note about the removal should still name it

    def test_combine_answer_applies_the_scrubber(self):
        # _combine_answer must scrub the intro even when no display_reports override is given.
        intro = "global_id 42: 7 confirmed interactions of their own -- invented."
        answer, _ = _combine_answer(intro, deterministic_reports=[])
        self.assertNotIn("42: 7 confirmed", answer)


class TestProximityReportMerging(unittest.TestCase):
    """Bug: an interactor's own confirmed-interaction count was rendered as
    a separate, disconnected count_nearby_entities(...) block the reader
    had to cross-reference by id themselves, instead of appearing on the
    same line as the interactor. Fixed with format_proximity_report_merged,
    which inlines the count only when a real count_nearby_entities result
    exists for that specific global_id."""

    def test_inlines_count_when_available(self):
        result = {
            "confirmed_proximity_matches": [
                {"camera": "537", "time": "0:01", "appearance_caption": "a man", "caption_agreement": 0.9,
                 "global_id": 5, "distance_m": 0.5},
            ]
        }
        count_results = {5: {"confirmed_interaction_count": 3}}
        report = format_proximity_report_merged(result, count_results)
        self.assertIn("3 confirmed interactions of their own", report)

    def test_no_count_suffix_when_not_yet_queried(self):
        result = {
            "confirmed_proximity_matches": [
                {"camera": "537", "time": "0:01", "appearance_caption": "a man", "caption_agreement": 0.9,
                 "global_id": 5, "distance_m": 0.5},
            ]
        }
        report = format_proximity_report_merged(result, count_results={})
        self.assertNotIn("confirmed interactions of their own", report)


class TestDisplayReportAssembly(unittest.TestCase):
    """Bug: the final answer's report assembly originally only reconstructed
    the rank_entities_by_interaction_count and find_nearby_entities/
    find_nearby_entities_by_description categories -- any OTHER
    deterministic-report tool's output (e.g. list_low_quality_caption_entities,
    called on its own with no preceding proximity/rank call) silently never
    reached the final answer at all, forcing the model to retype the whole
    thing itself from memory (and get cut off mid-list). Fixed by adding an
    explicit catch-all category (other_report_texts).

    Second bug, same area: rank_report_texts/other_report_texts were plain
    lists, so a no-argument tool (e.g. list_low_quality_caption_entities
    takes none, so every call returns the identical result) called several
    times in one conversation showed its ~70-entity report duplicated
    verbatim in the final answer. Fixed by keying both dicts by
    (name, call_args), same dedup pattern proximity_calls already used."""

    def test_other_report_tools_appear_in_final_display(self):
        other_report_texts = {("list_low_quality_caption_entities", ()):
                               "--- list_low_quality_caption_entities() ---\n30 entities flagged..."}
        reports = _build_display_reports(proximity_calls={}, rank_report_texts={},
                                          other_report_texts=other_report_texts, count_results={})
        self.assertIn(list(other_report_texts.values())[0], reports)

    def test_rank_and_proximity_and_other_all_present_together(self):
        rank_texts = {("rank_entities_by_interaction_count", ()):
                      "--- rank_entities_by_interaction_count() ---\ntop 3..."}
        other_texts = {("list_low_quality_caption_entities", ()):
                       "--- list_low_quality_caption_entities() ---\n30 entities..."}
        proximity_calls = {("find_nearby_entities", 7): (
            "--- find_nearby_entities(global_id=7) ---",
            {"target_global_id": 7, "confirmed_proximity_matches": []},
        )}
        reports = _build_display_reports(proximity_calls, rank_texts, other_texts, count_results={})
        joined = "\n".join(reports)
        self.assertIn("rank_entities_by_interaction_count", joined)
        self.assertIn("list_low_quality_caption_entities", joined)
        self.assertIn("find_nearby_entities", joined)

    def test_repeated_identical_call_does_not_duplicate_in_final_display(self):
        # The actual observed bug: the same no-argument tool called 3 times produced 3 copies of
        # its report. Simulates that by writing to the SAME key twice, as answer_query's loop does
        # for repeated identical calls -- the dict naturally collapses it to one entry.
        key = ("list_low_quality_caption_entities", ())
        other_report_texts = {}
        report_text = "--- list_low_quality_caption_entities() ---\n71 entities flagged..."
        other_report_texts[key] = report_text
        other_report_texts[key] = report_text  # second identical call, same key
        other_report_texts[key] = report_text  # third identical call, same key
        reports = _build_display_reports(proximity_calls={}, rank_report_texts={},
                                          other_report_texts=other_report_texts, count_results={})
        self.assertEqual(reports.count(report_text), 1)


class TestInventedIdRecovery(unittest.TestCase):
    """Bug: the model sometimes bundles a ranking/search call together with
    dependent detail calls in the SAME turn, before it could have seen the
    ranking result, and invents placeholder ids (e.g. 12345, 67890) to fill
    the gap. Those calls correctly error out at the tool layer, but the
    model then sometimes gave up ("not enough evidence") instead of using
    the real result sitting in the same batch. A system-prompt-only fix did
    not reliably prevent this; format_invented_id_recovery is the
    structural nudge that fires once an invented id actually errors."""

    def test_names_the_invented_ids_as_nonexistent(self):
        msg = format_invented_id_recovery([12345, 67890], known_entity_ids={7, 1, 9})
        self.assertIn("12345", msg)
        self.assertIn("67890", msg)
        self.assertIn("do not exist", msg)

    def test_names_the_real_ids_to_use_instead(self):
        msg = format_invented_id_recovery([12345], known_entity_ids={7, 1, 9})
        self.assertIn("7", msg)
        self.assertIn("1", msg)
        self.assertIn("9", msg)
        self.assertIn("REAL entity ids", msg)


class TestInteractorCountNudge(unittest.TestCase):
    """Bug: an earlier version sent one SEPARATE nudge per find_nearby_entities
    call in the same turn (e.g. 3 simultaneous calls -> 3 competing nudges),
    which caused the model to conflate the id lists and re-query the
    original 3 entities instead of their actual interactors. Fixed by
    consolidating into exactly one nudge per iteration, covering every
    call made in that batch."""

    def test_nudge_names_all_batch_ids(self):
        msg = format_interactor_count_nudge([542, 169, 15], remaining_after=0)
        for gid in (542, 169, 15):
            self.assertIn(str(gid), msg)

    def test_nudge_mentions_remaining_when_nonzero(self):
        msg = format_interactor_count_nudge([1, 2, 3], remaining_after=12)
        self.assertIn("12", msg)
        self.assertIn("more", msg)

    def test_nudge_says_nothing_about_remaining_when_zero(self):
        msg = format_interactor_count_nudge([1, 2, 3], remaining_after=0)
        # Should not claim there are more batches coming when the queue is empty.
        self.assertNotIn("more after this batch", msg)

    def test_empty_batch_produces_empty_string(self):
        self.assertEqual(format_interactor_count_nudge([], remaining_after=0), "")


class TestNarrationWithoutAction(unittest.TestCase):
    """Bug: the model sometimes described calling a tool in prose/markdown
    ("I need to call find_nearby_entities for global_id 7...") without
    actually emitting a real <tool_call> block. _mentions_unexecuted_tool
    is the detector that catches this so the loop can nudge a retry instead
    of accepting the narration as a final answer."""

    def test_detects_narrated_tool_name(self):
        text = "I need to call find_nearby_entities for global_id 7 to check this."
        self.assertEqual(_mentions_unexecuted_tool(text, ["find_nearby_entities", "count_nearby_entities"]),
                          "find_nearby_entities")

    def test_returns_none_for_ordinary_prose(self):
        text = "Based on the evidence, this entity was seen in camera 537 at 0:05."
        self.assertIsNone(_mentions_unexecuted_tool(text, ["find_nearby_entities", "count_nearby_entities"]))


class TestToolCallExtraction(unittest.TestCase):
    """Sanity coverage for _extract_tool_calls, the parser every other fix
    in this file depends on being correct -- including tolerating a
    malformed block (the "Ronaldo" garbling case) without crashing, only
    silently skipping the one block it can't parse."""

    def test_extracts_single_call(self):
        text = '<tool_call>\n{"name": "find_nearby_entities", "arguments": {"global_id": 7}}\n</tool_call>'
        calls = _extract_tool_calls(text)
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["name"], "find_nearby_entities")

    def test_extracts_multiple_calls_in_one_turn(self):
        text = (
            '<tool_call>\n{"name": "count_nearby_entities", "arguments": {"global_id": 1}}\n</tool_call>\n'
            '<tool_call>\n{"name": "count_nearby_entities", "arguments": {"global_id": 2}}\n</tool_call>'
        )
        calls = _extract_tool_calls(text)
        self.assertEqual(len(calls), 2)

    def test_malformed_block_is_skipped_not_fatal(self):
        # Simulates the observed "Ronaldo" garbling: a real block followed by one where the model's
        # generation degraded into an unparseable non-JSON body instead of the expected tag/JSON pair.
        text = (
            '<tool_call>\n{"name": "count_nearby_entities", "arguments": {"global_id": 1}}\n</tool_call>\n'
            'Ronaldo\n{"name": "count_nearby_entities", "arguments": {"global_id": 2}}\n</tool_call>'
        )
        calls = _extract_tool_calls(text)  # must not raise
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["arguments"]["global_id"], 1)

    def test_no_calls_returns_empty_list(self):
        self.assertEqual(_extract_tool_calls("just ordinary text, no tool calls here"), [])


class TestDeterministicReportsNeverRetyped(unittest.TestCase):
    """Bug: the model's own free text repeatedly corrupted facts it was
    given verbatim (miscounting a list, bleeding one entity's caption onto
    another). The fix moved every fact-bearing tool result into a
    deterministic, code-formatted report appended VERBATIM after the
    model's short framing text, which _combine_answer never lets the model
    override, retype, or omit."""

    def test_deterministic_reports_appear_verbatim_in_final_answer(self):
        report = "--- rank_entities_by_interaction_count() ---\nglobal_id 7: 40 confirmed interactions"
        answer, source = _combine_answer("Here is the ranking:", deterministic_reports=[], display_reports=[report])
        self.assertIn(report, answer)
        self.assertEqual(source, "llm_framing_plus_deterministic_reports")

    def test_no_reports_falls_back_to_free_text(self):
        answer, source = _combine_answer("I don't have enough evidence.", deterministic_reports=[])
        self.assertEqual(answer, "I don't have enough evidence.")
        self.assertEqual(source, "llm_free_text")

    def test_display_reports_can_differ_in_formatting_from_verification_source(self):
        # display_reports (what's shown) and deterministic_reports (what's checked against for
        # fabrication) are allowed to differ in formatting without affecting which facts are trusted --
        # this is what lets format_proximity_report_merged show a nicer view without breaking the scrubber.
        raw_report = "global_id 5: 3 confirmed interactions of their own -- a man."
        pretty_report = "  - global_id 5, 0.5m away, 3 confirmed interactions of their own"
        intro = "global_id 5: 3 confirmed interactions of their own -- fabricated retelling."
        answer, _ = _combine_answer(intro, deterministic_reports=[raw_report], display_reports=[pretty_report])
        self.assertIn(pretty_report, answer)
        self.assertIn("5: 3 confirmed", answer)  # the intro's claim about id 5 is verified, so kept


class TestCaptionConfidenceThresholdCalibration(unittest.TestCase):
    """Bug (BLIP era): an initial guess of 0.5 was checked against the real distribution of 2063
    sampled BLIP captions (mean 0.412, median 0.409, p90 0.508) and found to exclude 88% of ALL
    captions, not just outliers -- effectively disabling the multi-crop consensus mechanism for
    nearly every entity. Recalibrated to 0.3 (~p10 of BLIP's distribution).

    Renamed BLIP_CONFIDENCE_THRESHOLD -> CAPTION_CONFIDENCE_THRESHOLD when the captioning model
    switched to Qwen2.5-VL-3B (captions/matching/actions work) -- re-checked against Qwen's real
    distribution (193+ real POM captions: min=0.921, p10=0.975, mean=0.992) and found NOT to
    transfer as a useful signal at all (no meaningful low-confidence tail for this model's short,
    templated task). Deliberately left at 0.3 anyway, as an effectively-inert safety net rather
    than a real per-model recalibration -- this test guards against a FUTURE accidental push into
    the 0.9-1.0 range, which WOULD start excluding real captions again (repeating the original
    88%-exclusion regression, just via a different, backwards route)."""

    def test_threshold_is_not_the_known_bad_blip_value(self):
        self.assertNotEqual(CAPTION_CONFIDENCE_THRESHOLD, 0.5)

    def test_threshold_stays_in_the_inert_safety_net_range(self):
        # Must stay well below Qwen2.5-VL-3B's real observed range (0.921-1.0) or it would start
        # excluding real, good captions -- the exact class of regression this constant's whole
        # history is about.
        self.assertGreater(CAPTION_CONFIDENCE_THRESHOLD, 0.0)
        self.assertLess(CAPTION_CONFIDENCE_THRESHOLD, 0.9)


class TestCaptionAgreementThresholdConsistency(unittest.TestCase):
    """Sanity: the threshold used to decide "low confidence" must be shared
    between agreement_flag and format_low_quality_caption_report's
    low/high split -- a drift between the two would make the split
    inconsistent with the inline [caption uncertain] flag."""

    def test_threshold_is_the_documented_value(self):
        self.assertEqual(CAPTION_AGREEMENT_THRESHOLD, 0.6)


class TestColorWordStripping(unittest.TestCase):
    """Bug: strip_color_words' first version only matched whole tokens against COLOR_STOPWORDS, so
    hyphenated compounds like "dark-colored jacket" sailed straight through untouched (neither
    "dark-colored" nor "dark" nor "colored" is itself in the stopword set as a standalone token
    match). Real, observed leak in extract_entity_attributes.py output: "The person is wearing a
    dark-colored jacket." Fixed by also splitting each token on '-' and checking its parts."""

    def test_strips_plain_color_word(self):
        self.assertEqual(strip_color_words("a black jacket"), "a jacket")

    def test_strips_hyphenated_color_compound(self):
        result = strip_color_words("a dark-colored jacket")
        self.assertNotIn("dark-colored", result)
        self.assertIn("jacket", result)

    def test_strips_hyphenated_compound_at_start(self):
        result = strip_color_words("light-colored pants and a red-and-white scarf")
        self.assertNotIn("light-colored", result)
        self.assertNotIn("red-and-white", result)
        self.assertIn("pants", result)
        self.assertIn("scarf", result)

    def test_leaves_non_color_words_untouched(self):
        self.assertEqual(strip_color_words("a jacket and jeans"), "a jacket and jeans")


class TestDominantColorNaming(unittest.TestCase):
    """dominant_color_name should recover a color close to a synthetic solid-color region's actual
    RGB value -- a basic sanity check that the k-means + nearest-Lab-distance pipeline isn't
    accidentally inverted or channel-swapped (the exact class of bug that once made every ReID crop's
    color signal silently wrong via an unflipped BGR/RGB mismatch elsewhere in this project)."""

    def test_solid_black_region_is_named_black(self):
        region_rgb = np.full((40, 40, 3), NAMED_COLORS_RGB["black"], dtype=np.uint8)
        region_bgr = region_rgb[:, :, ::-1]
        self.assertEqual(dominant_color_name(region_bgr), "black")

    def test_solid_red_region_is_named_red_not_blue(self):
        # Specifically checks channel order isn't swapped -- a red region should never be named
        # "blue" (or vice versa), which is exactly what a BGR/RGB mixup would produce.
        region_rgb = np.full((40, 40, 3), NAMED_COLORS_RGB["red"], dtype=np.uint8)
        region_bgr = region_rgb[:, :, ::-1]
        name = dominant_color_name(region_bgr)
        self.assertNotEqual(name, "blue")
        self.assertNotEqual(name, "dark blue")

    def test_empty_region_returns_unknown(self):
        self.assertEqual(dominant_color_name(np.zeros((0, 0, 3), dtype=np.uint8)), "unknown")


class TestCropQualityGateCalibration(unittest.TestCase):
    """Bug: the first crop-quality gate was calibrated against a RANDOM (camera, frame) sample from
    pred_full.txt (p5 area=1135) -- a different, larger-skewing population than what
    extract_entity_attributes.py actually samples (crops spread across each entity's own
    trajectory, real p5 area=364). Applied blindly, this flagged 41% of all 934 real POM entities
    as "unclear" instead of a real ~8%. Recalibrated to area-only (>=400px, from direct visual
    inspection of real confirmed-bad ~266-290px examples vs. a confirmed-legible ~624px one),
    dropping a WIDTH-only rule that had separately, incorrectly rejected a legible 16x185px crop
    (narrow only because the person was captured in a tight upright bbox)."""

    def test_tiny_area_crop_is_flagged_low_quality(self):
        # 10x20 = 200px area, well below MIN_CROP_AREA_PX -- a real confirmed-bad example was
        # 266-290px area.
        crop = np.random.randint(0, 255, (20, 10, 3), dtype=np.uint8)
        self.assertTrue(is_low_quality_crop(crop))

    def test_narrow_but_tall_crop_is_not_rejected_by_width_alone(self):
        # Regression for the width-only false rejection: 16px wide x 185px tall (area=2960, well
        # above threshold) was a real, legible crop the first version wrongly rejected. Uses varied
        # pixel values (not flat) so the Laplacian-variance blur check also passes.
        crop = np.random.randint(0, 255, (185, 16, 3), dtype=np.uint8)
        self.assertFalse(is_low_quality_crop(crop))

    def test_area_threshold_is_between_confirmed_bad_and_confirmed_good_examples(self):
        # 266-290px (confirmed bad via direct visual inspection) < threshold < 624px (confirmed
        # legible) -- guards against the threshold drifting back toward either extreme.
        self.assertGreater(MIN_CROP_AREA_PX, 290)
        self.assertLess(MIN_CROP_AREA_PX, 624)

    def test_empty_crop_is_flagged_low_quality(self):
        self.assertTrue(is_low_quality_crop(np.zeros((0, 0, 3), dtype=np.uint8)))


class TestSameCameraOverlapVeto(unittest.TestCase):
    """The single highest-risk detail identified before implementing within-camera identity
    stitching: plausibility_matrix's cross-camera overlap rule only vetoes overlapping tracks that
    are ALSO far apart (dist > OVERLAP_DISTANCE_TOLERANCE_M=3.0) -- correct cross-camera (two
    cameras can legitimately see the same person from different angles at once), but WRONG if reused
    unchanged for same-camera pairs, where two simultaneously-visible boxes are always two different
    people regardless of distance. plausibility_matrix's same_camera=True branch must give an
    ABSOLUTE veto (any overlap -> 0), not the distance-conditional cross-camera one."""

    def _overlapping_close_tracks(self):
        # Two tracks active in the exact same frame window, 1m apart -- well within the 3m
        # cross-camera tolerance, so a same_camera=False call should NOT veto this pair.
        first_a, last_a = np.array([100]), np.array([200])
        pos_a = np.array([[0.0, 0.0]])
        first_b, last_b = np.array([100]), np.array([200])
        pos_b = np.array([[1.0, 0.0]])
        return first_a, last_a, pos_a, first_b, last_b, pos_b

    def test_same_camera_overlap_is_hard_vetoed_regardless_of_close_distance(self):
        args = self._overlapping_close_tracks()
        plaus = plausibility_matrix(*args, same_camera=True)
        self.assertEqual(plaus[0, 0], 0.0)

    def test_cross_camera_overlap_within_tolerance_is_not_vetoed(self):
        # Real finding while writing this test, not previously documented: the explicit "overlap
        # AND dist > 3.0m -> 0" veto is almost redundant for cross-camera pairs, because the
        # GENERIC required_speed=dist/gap_sec formula already drives plausibility to ~0 for any
        # overlapping pair beyond ~0.1m (gap_sec is floored at 1 frame for an overlapping pair, so
        # even a modest distance implies an enormous required speed). The 1m-apart case above
        # (within the 3.0m explicit-veto tolerance) still lands at exactly 0 via this generic path,
        # not the hard veto -- only genuinely near-zero distances get real credit during overlap.
        # Not changed here (it's pre-existing behavior, not something this session's within-camera
        # work touched, and changing it would affect already-reported quantitative baselines) --
        # flagged honestly instead of silently asserting around it.
        first_a, last_a = np.array([100]), np.array([200])
        pos_a = np.array([[0.0, 0.0]])
        first_b, last_b = np.array([100]), np.array([200])
        pos_b = np.array([[0.02, 0.0]])  # 2cm apart -- genuinely near-zero distance
        plaus = plausibility_matrix(first_a, last_a, pos_a, first_b, last_b, pos_b, same_camera=False)
        self.assertGreater(plaus[0, 0], 0.0)

    def test_cross_camera_overlap_beyond_tolerance_is_still_vetoed(self):
        first_a, last_a = np.array([100]), np.array([200])
        pos_a = np.array([[0.0, 0.0]])
        first_b, last_b = np.array([100]), np.array([200])
        pos_b = np.array([[10.0, 0.0]])  # 10m apart, well beyond the 3m tolerance
        plaus = plausibility_matrix(first_a, last_a, pos_a, first_b, last_b, pos_b, same_camera=False)
        self.assertEqual(plaus[0, 0], 0.0)


class TestFrameRangeWideningRegression(unittest.TestCase):
    """Bug: cross_camera_reid.py's within-camera group consolidation originally widened a group's
    frame range to min/max across ALL members, which artificially inflated a group's temporal
    footprint and spuriously triggered the overlap branch above against unrelated cross-camera
    candidates -- collapsing real spanning matches from 121 to 13 in the first version. This test
    encodes the underlying mechanism directly: the SAME pair of tracks must be plausible (a
    legitimate sequential, walkable-gap candidate) when using a TIGHT frame range, but implausible
    once one track's window is artificially widened to overlap the other -- confirming why the fix
    (using a group representative's own tight range, not a widened envelope) matters."""

    def test_tight_frame_ranges_yield_plausible_sequential_match(self):
        # Track A ends at frame 100, track B starts at frame 130 (1 second gap @ 30fps) -- an easy
        # walking-pace gap for two positions 1m apart.
        first_a, last_a, pos_a = np.array([50]), np.array([100]), np.array([[0.0, 0.0]])
        first_b, last_b, pos_b = np.array([130]), np.array([180]), np.array([[1.0, 0.0]])
        plaus = plausibility_matrix(first_a, last_a, pos_a, first_b, last_b, pos_b, same_camera=False)
        self.assertGreater(plaus[0, 0], 0.5)

    def test_widened_frame_range_spuriously_collapses_the_same_pair(self):
        # Same two tracks, but track A's window is now widened (as the buggy consolidation once
        # did) to span all the way to frame 200 -- now overlapping track B's window entirely, even
        # though the underlying physical encounter is unchanged.
        first_a, last_a, pos_a = np.array([50]), np.array([200]), np.array([[0.0, 0.0]])
        first_b, last_b, pos_b = np.array([130]), np.array([180]), np.array([[1.0, 0.0]])
        plaus = plausibility_matrix(first_a, last_a, pos_a, first_b, last_b, pos_b, same_camera=False)
        self.assertLess(plaus[0, 0], 0.5)


class TestCaptionSimilarityAdditiveCombination(unittest.TestCase):
    """match_pairs combines appearance and caption similarity ADDITIVELY
    (w_appearance * appearance_sim + w_caption * caption_sim), not as another hard multiplicative
    gate, and falls back to appearance-only for any pair missing a caption embedding rather than
    penalizing missing data. Verified via real POM evaluation that a non-trivial w_caption can
    increase over-merging risk on a population with generic clothing (see CLAUDE.md) -- kept
    disabled by default there, but the combination math itself must stay correct regardless."""

    def _two_track_setup(self, with_captions: bool):
        keys = [("cam0", 1), ("cam0", 2)]
        embeddings = {("cam0", 1): np.array([1.0, 0.0]), ("cam0", 2): np.array([1.0, 0.0])}
        world_pos = {k: np.array([0.0, 0.0]) for k in keys}
        first_frame = {("cam0", 1): 0, ("cam0", 2): 10000}  # far apart in time -> full plausibility
        last_frame = {("cam0", 1): 5, ("cam0", 2): 10005}
        caption_embeddings = None
        if with_captions:
            # Identical appearance, but caption embeddings anti-correlated -- if the additive
            # combination is working, this should pull the combined score DOWN from pure appearance.
            caption_embeddings = {("cam0", 1): np.array([1.0, 0.0]), ("cam0", 2): np.array([-1.0, 0.0])}
        return keys, embeddings, world_pos, first_frame, last_frame, caption_embeddings

    def test_missing_caption_embeddings_falls_back_to_appearance_only(self):
        keys, embeddings, world_pos, first_frame, last_frame, _ = self._two_track_setup(with_captions=False)
        pairs = match_pairs([keys[0]], [keys[1]], embeddings, world_pos, first_frame, last_frame,
                             threshold=0.99, same_camera=False, caption_embeddings=None)
        # Appearance similarity is 1.0 (identical vectors) and plausibility is full -> should match
        # at a threshold just below 1.0.
        self.assertEqual(pairs, [(keys[0], keys[1])])

    def test_caption_similarity_pulls_combined_score_down_when_anti_correlated(self):
        keys, embeddings, world_pos, first_frame, last_frame, caption_embeddings = self._two_track_setup(with_captions=True)
        # Same appearance (sim=1.0) as the fallback case, but now anti-correlated captions
        # (cap_sim=-1.0) with w_caption=0.5 should pull the combined score down to 0.5*1.0 +
        # 0.5*(-1.0) = 0.0, well below a high threshold that passed in the no-caption case above.
        pairs = match_pairs([keys[0]], [keys[1]], embeddings, world_pos, first_frame, last_frame,
                             threshold=0.99, same_camera=False, caption_embeddings=caption_embeddings,
                             w_appearance=0.5, w_caption=0.5)
        self.assertEqual(pairs, [])


class TestSameCameraOverlapTransitiveInvariant(unittest.TestCase):
    """Real user-raised concern: the same-camera hard veto in plausibility_matrix only guards a
    SINGLE Hungarian call's own direct matches. During the cross-camera pass, group A could match
    group C via camera-pair (0,2), and group B could separately match group C via camera-pair
    (1,2) -- two independent, individually-valid Hungarian results that transitively fuse A and B
    through their shared root, even though A and B were never compared directly. If A and B both
    contain same-camera-overlapping raw tracks (a real physical impossibility), that fusion would
    silently reintroduce it. merge_would_violate_same_camera_overlap must catch this."""

    def test_detects_violation_introduced_via_third_group(self):
        # Track A (cam0, frames 0-100) and track B (cam0, frames 50-150) overlap in the SAME
        # camera -- two different physical people, by construction. Track C (cam1) already got
        # unioned with A (simulating an earlier, independent cross-camera match).
        all_keys = [("cam0", 1), ("cam0", 2), ("cam1", 1)]
        first_frame = {("cam0", 1): 0, ("cam0", 2): 50, ("cam1", 1): 200}
        last_frame = {("cam0", 1): 100, ("cam0", 2): 150, ("cam1", 1): 300}
        uf = UnionFind(all_keys)
        uf.union(("cam0", 1), ("cam1", 1))  # A and C already fused (an earlier valid cross-camera match)
        # Now checking whether B can also join C's group -- it must NOT be allowed, since that
        # would transitively put A (cam0, 0-100) and B (cam0, 50-150) in the same entity.
        violates = merge_would_violate_same_camera_overlap(
            all_keys, uf, ("cam0", 2), ("cam1", 1), first_frame, last_frame)
        self.assertTrue(violates)

    def test_allows_merge_with_no_overlap(self):
        all_keys = [("cam0", 1), ("cam0", 2), ("cam1", 1)]
        first_frame = {("cam0", 1): 0, ("cam0", 2): 500, ("cam1", 1): 200}  # cam0 tracks don't overlap
        last_frame = {("cam0", 1): 100, ("cam0", 2): 600, ("cam1", 1): 300}
        uf = UnionFind(all_keys)
        uf.union(("cam0", 1), ("cam1", 1))
        violates = merge_would_violate_same_camera_overlap(
            all_keys, uf, ("cam0", 2), ("cam1", 1), first_frame, last_frame)
        self.assertFalse(violates)


class TestGarmentRegionMinimumSize(unittest.TestCase):
    """Real bug, user-reported: a 38x13px crop (494px total area -- large enough to pass
    crop_quality's coarser area>=400 gate) sliced down to a ~72px garment region gave "black" for
    a crop that was genuinely cream/beige (confirmed by direct visual inspection); the SAME
    person's larger crops (580px+ garment region) correctly gave "beige". crop_quality's gate
    operates on the WHOLE crop, not garment_region()'s sub-slice, so a crop can pass it and still
    leave too few pixels for k-means to find a real color cluster. dominant_color_name must return
    "unknown" rather than guess when the region is too small."""

    def test_tiny_region_returns_unknown_not_a_guess(self):
        # ~72px, matching the real confirmed-bad case's approximate scale.
        region = np.random.randint(0, 255, (9, 8, 3), dtype=np.uint8)
        self.assertEqual(dominant_color_name(region), "unknown")

    def test_adequately_sized_region_is_not_forced_to_unknown(self):
        region = np.full((30, 30, 3), NAMED_COLORS_RGB["beige"], dtype=np.uint8)
        self.assertNotEqual(dominant_color_name(region), "unknown")

    def test_threshold_is_below_the_real_legitimate_population(self):
        # Calibrated against a real 18-crop sample's garment-region sizes (p5=450px) -- must stay
        # comfortably below that or legitimate crops would start getting "unknown" too.
        self.assertLess(MIN_GARMENT_REGION_PIXELS, 450)
        self.assertGreater(MIN_GARMENT_REGION_PIXELS, 72)


class TestSameCameraBoundaryTouchNotVetoed(unittest.TestCase):
    """Real bug found via direct visual inspection (user-raised: "only ~10 people, must be a bug"):
    a same-camera track pair touching at the EXACT boundary frame (track A's last frame == track
    B's first frame -- a classic ID-switch handoff, confirmed concretely on POM camera_0000 tracks
    419/488, literally the same detection at the same frame) was being hard-vetoed by the same-
    camera overlap rule's `first_b <= last_a` check, which is True at a zero-duration touch. Fixed
    by requiring overlap DURATION to exceed SAME_CAMERA_OVERLAP_GRACE_FRAMES before vetoing."""

    def test_exact_boundary_touch_is_not_vetoed(self):
        first_a, last_a, pos_a = np.array([100]), np.array([300]), np.array([[0.0, 0.0]])
        first_b, last_b, pos_b = np.array([300]), np.array([500]), np.array([[0.0, 0.0]])
        plaus = plausibility_matrix(first_a, last_a, pos_a, first_b, last_b, pos_b, same_camera=True)
        self.assertGreater(plaus[0, 0], 0.0)

    def test_overlap_beyond_grace_window_is_still_vetoed(self):
        # Real overlap duration well beyond the grace window -- must still be a hard veto.
        first_a, last_a, pos_a = np.array([100]), np.array([300]), np.array([[0.0, 0.0]])
        first_b, last_b, pos_b = np.array([150]), np.array([350]), np.array([[0.0, 0.0]])
        overlap_duration = min(300, 350) - max(100, 150)
        self.assertGreater(overlap_duration, SAME_CAMERA_OVERLAP_GRACE_FRAMES)
        plaus = plausibility_matrix(first_a, last_a, pos_a, first_b, last_b, pos_b, same_camera=True)
        self.assertEqual(plaus[0, 0], 0.0)


class TestCleanHandoffOverridesNoisyAppearance(unittest.TestCase):
    """Real bug found via direct visual inspection: several confirmed real same-person handoffs had
    appearance similarity as low as 0.59 (motion blur/tiny-crop noise), which no reasonable
    appearance-only threshold could accept without also accepting false positives elsewhere. A
    "clean handoff" (tight time gap AND tight pixel-space displacement) should OVERRIDE a noisy
    appearance score for same-camera matching. Two sub-bugs guarded here: the confidence formula
    must give FULL credit within the calibrated range (not decay linearly from zero), and the
    handoff score must apply AFTER plausibility multiplication, not be silently zeroed by the same
    unreliable world-position-based plausibility it exists to route around."""

    def _setup(self, appearance_sim: float, gap_frames: int, px_dist: float):
        keys = [("cam0", 1), ("cam0", 2)]
        # Anti-correlated-ish embeddings scaled to produce the desired cosine similarity isn't
        # trivial with raw vectors; use two 2D unit vectors at a controlled angle instead.
        theta = np.arccos(np.clip(appearance_sim, -1, 1))
        embeddings = {("cam0", 1): np.array([1.0, 0.0]), ("cam0", 2): np.array([np.cos(theta), np.sin(theta)])}
        first_frame = {("cam0", 1): 100, ("cam0", 2): 100 + gap_frames}
        last_frame = {("cam0", 1): 100, ("cam0", 2): 100 + gap_frames + 50}
        # World position kept identical (irrelevant/misleading on purpose -- the point is the
        # handoff signal must not depend on it) but far apart, to confirm the override bypasses a
        # bad plausibility reading, matching the real diagnosed case.
        world_pos = {("cam0", 1): np.array([0.0, 0.0]), ("cam0", 2): np.array([900.0, 900.0])}
        first_pixel_pos = {("cam0", 1): np.array([0.0, 0.0]), ("cam0", 2): np.array([px_dist, 0.0])}
        last_pixel_pos = {("cam0", 1): np.array([0.0, 0.0]), ("cam0", 2): np.array([px_dist, 0.0])}
        return keys, embeddings, world_pos, first_frame, last_frame, first_pixel_pos, last_pixel_pos

    def test_clean_handoff_rescues_low_appearance_similarity(self):
        keys, embeddings, world_pos, first_frame, last_frame, first_pixel_pos, last_pixel_pos = self._setup(
            appearance_sim=0.59, gap_frames=6, px_dist=20)
        pairs = match_pairs([keys[0]], [keys[1]], embeddings, world_pos, first_frame, last_frame,
                             threshold=0.75, same_camera=True,
                             first_pixel_pos=first_pixel_pos, last_pixel_pos=last_pixel_pos)
        self.assertEqual(pairs, [(keys[0], keys[1])])

    def test_far_handoff_does_not_rescue_low_appearance_similarity(self):
        keys, embeddings, world_pos, first_frame, last_frame, first_pixel_pos, last_pixel_pos = self._setup(
            appearance_sim=0.59, gap_frames=200, px_dist=500)
        pairs = match_pairs([keys[0]], [keys[1]], embeddings, world_pos, first_frame, last_frame,
                             threshold=0.75, same_camera=True,
                             first_pixel_pos=first_pixel_pos, last_pixel_pos=last_pixel_pos)
        self.assertEqual(pairs, [])

    def test_no_pixel_position_data_falls_back_gracefully(self):
        keys, embeddings, world_pos, first_frame, last_frame, _, _ = self._setup(
            appearance_sim=0.59, gap_frames=6, px_dist=20)
        # No crash, no phantom match, when pixel position data isn't available.
        pairs = match_pairs([keys[0]], [keys[1]], embeddings, world_pos, first_frame, last_frame,
                             threshold=0.75, same_camera=True)
        self.assertEqual(pairs, [])


class TestIterativeStitchingResolvesChains(unittest.TestCase):
    """Real bug found via direct visual inspection: a single Hungarian round enforces at most one
    match per track, so a person fragmented into 3+ pieces could only pair up floor(n/2) of them per
    round even with every pairwise similarity above threshold (confirmed: 419 matched to 749, 488
    matched to 928 in round 1, even though 419-488 themselves scored above threshold). This test
    encodes the mechanism directly: consolidate_group_representatives, called iteratively across
    rounds via a shared UnionFind, must eventually fuse a 4-track chain that a single round cannot."""

    def test_four_track_chain_resolves_after_two_rounds(self):
        # A: matches C well; B: matches D well; but A-C and B-D are the round-1 Hungarian optimum,
        # leaving the true A-B link (via C/D's consolidated representatives) for round 2.
        keys = [("cam0", "A"), ("cam0", "B"), ("cam0", "C"), ("cam0", "D")]
        embeddings = {
            ("cam0", "A"): np.array([1.0, 0.0]),
            ("cam0", "C"): np.array([0.99, np.sqrt(1 - 0.99 ** 2)]),  # A's best round-1 match
            ("cam0", "B"): np.array([0.0, 1.0]) * -1 + np.array([1.0, 0.0]) * 0.8,  # placeholder, overwritten below
            ("cam0", "D"): np.array([1.0, 0.0]),
        }
        # Simpler, explicit construction: A and C are near-identical (best round-1 pair); B and D
        # are near-identical (the other round-1 pair); A/B are similar enough to merge once C/D's
        # groups are each represented by a single point close to A and B respectively.
        embeddings[("cam0", "A")] = np.array([1.0, 0.0])
        embeddings[("cam0", "C")] = np.array([0.999, np.sqrt(1 - 0.999 ** 2)])
        embeddings[("cam0", "B")] = np.array([0.97, np.sqrt(1 - 0.97 ** 2)])
        embeddings[("cam0", "D")] = np.array([0.971, np.sqrt(1 - 0.971 ** 2)])
        world_pos = {k: np.array([0.0, 0.0]) for k in keys}
        # Non-overlapping, walkable-gap frame ranges for every pair (dist=0 everywhere -> full
        # plausibility regardless of gap).
        first_frame = {("cam0", "A"): 0, ("cam0", "C"): 10000, ("cam0", "B"): 20000, ("cam0", "D"): 30000}
        last_frame = {("cam0", "A"): 5, ("cam0", "C"): 10005, ("cam0", "B"): 20005, ("cam0", "D"): 30005}

        uf = UnionFind(keys)
        threshold = 0.9
        for _ in range(5):
            reps, group_members = consolidate_group_representatives(keys, uf, embeddings, first_frame, last_frame, world_pos)
            pairs = match_pairs(reps, reps, embeddings, world_pos, first_frame, last_frame,
                                 threshold, same_camera=True)
            if not pairs:
                break
            for a, b in pairs:
                uf.union(a, b)

        roots = {k: uf.find(k) for k in keys}
        # All 4 should end up in the same group after enough rounds, even though A-B and C-D were
        # never each other's best single-round match.
        self.assertEqual(len(set(roots.values())), 1, f"expected all 4 keys in one group, got {roots}")


if __name__ == "__main__":
    unittest.main()
