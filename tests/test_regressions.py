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

from extract_entity_attributes import is_degenerate_caption, BLIP_CONFIDENCE_THRESHOLD
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
    explicit catch-all category (other_report_texts)."""

    def test_other_report_tools_appear_in_final_display(self):
        other_report_texts = ["--- list_low_quality_caption_entities() ---\n30 entities flagged..."]
        reports = _build_display_reports(proximity_calls={}, rank_report_texts=[],
                                          other_report_texts=other_report_texts, count_results={})
        self.assertIn(other_report_texts[0], reports)

    def test_rank_and_proximity_and_other_all_present_together(self):
        rank_texts = ["--- rank_entities_by_interaction_count() ---\ntop 3..."]
        other_texts = ["--- list_low_quality_caption_entities() ---\n30 entities..."]
        proximity_calls = {("find_nearby_entities", 7): (
            "--- find_nearby_entities(global_id=7) ---",
            {"target_global_id": 7, "confirmed_proximity_matches": []},
        )}
        reports = _build_display_reports(proximity_calls, rank_texts, other_texts, count_results={})
        joined = "\n".join(reports)
        self.assertIn("rank_entities_by_interaction_count", joined)
        self.assertIn("list_low_quality_caption_entities", joined)
        self.assertIn("find_nearby_entities", joined)


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


class TestBlipConfidenceThresholdCalibration(unittest.TestCase):
    """Bug: an initial guess of BLIP_CONFIDENCE_THRESHOLD=0.5 was checked
    against the real distribution of 2063 sampled captions (mean 0.412,
    median 0.409, p90 0.508) and found to exclude 88% of ALL captions, not
    just outliers -- effectively disabling the multi-crop consensus
    mechanism for nearly every entity. Recalibrated to 0.3 (~p10). This
    test doesn't re-derive the distribution (that needs the real dataset --
    see test_graph_integration.py) but guards against the threshold being
    silently pushed back into the range that was empirically shown broken."""

    def test_threshold_is_not_the_known_bad_value(self):
        self.assertNotEqual(BLIP_CONFIDENCE_THRESHOLD, 0.5)

    def test_threshold_is_in_the_calibrated_range(self):
        # Calibrated against p10 (~0.316) of the real distribution -- anything meaningfully above
        # that would repeat the 88%-exclusion regression; anything at 0 would disable the check.
        self.assertGreater(BLIP_CONFIDENCE_THRESHOLD, 0.0)
        self.assertLess(BLIP_CONFIDENCE_THRESHOLD, 0.4)


class TestCaptionAgreementThresholdConsistency(unittest.TestCase):
    """Sanity: the threshold used to decide "low confidence" must be shared
    between agreement_flag and format_low_quality_caption_report's
    low/high split -- a drift between the two would make the split
    inconsistent with the inline [caption uncertain] flag."""

    def test_threshold_is_the_documented_value(self):
        self.assertEqual(CAPTION_AGREEMENT_THRESHOLD, 0.6)


if __name__ == "__main__":
    unittest.main()
