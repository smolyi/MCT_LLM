"""
SLOW, GPU-dependent smoke tests that load the real Qwen2.5-7B-Instruct
model and run real generation against the real event graph -- these are
the only tests in this project that exercise actual LLM behavior, as
opposed to the deterministic Python code around it (test_regressions.py)
or the graph data itself (test_graph_integration.py).

Skipped by default (model loading alone takes real time, and each query
here can take minutes -- this session's own debugging repeatedly involved
multi-minute background runs for exactly these kinds of checks). Run
explicitly with:

  RUN_LLM_TESTS=1 python tests/test_llm_smoke.py -v

These use greedy decoding (do_sample=False, same as query_interface.py's
default), so a given query's output is deterministic given identical code
-- that's what makes "did this regression come back" checkable at all, and
is the same property this project's whole debugging methodology this
session relied on (re-run an identical query after one code change,
attribute any behavior difference to that change).

The model is loaded ONCE via setUpModule/tearDownModule (module-level
fixtures), not per-TestCase setUpClass -- a real bug in an earlier version
of this file gave each TestCase its own setUpClass, which unittest calls
once per class, silently loading the ~15GB model a second time before the
first one was released and causing a genuine CUDA OOM.
"""
import os
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

GRAPH_PATH = PROJECT_ROOT / "data" / "scene_061" / "event_graph_with_attrs.gpickle"
RUN_LLM_TESTS = os.environ.get("RUN_LLM_TESTS") == "1"

_state = {}  # populated by setUpModule: tools, tool_fns, tokenizer, model


def setUpModule():
    if not (RUN_LLM_TESTS and GRAPH_PATH.exists()):
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from graph_tools import GraphTools

    tools = GraphTools(str(GRAPH_PATH))
    _state["tools"] = tools
    _state["tool_fns"] = {
        "search_by_appearance": tools.search_by_appearance,
        "get_entity_timeline": tools.get_entity_timeline,
        "find_entities_in_camera": tools.find_entities_in_camera,
        "check_entities_cooccur": tools.check_entities_cooccur,
        "find_nearby_entities": tools.find_nearby_entities,
        "count_nearby_entities": tools.count_nearby_entities,
        "find_nearby_entities_by_description": tools.find_nearby_entities_by_description,
        "list_multi_camera_entities": tools.list_multi_camera_entities,
        "list_low_quality_caption_entities": tools.list_low_quality_caption_entities,
        "rank_entities_by_interaction_count": tools.rank_entities_by_interaction_count,
    }
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    _state["tokenizer"] = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    _state["model"] = model


def tearDownModule():
    if "model" in _state:
        import torch
        del _state["model"]
        torch.cuda.empty_cache()


@unittest.skipUnless(RUN_LLM_TESTS, "set RUN_LLM_TESTS=1 to run slow, GPU-dependent LLM smoke tests")
@unittest.skipUnless(GRAPH_PATH.exists(), f"requires the real graph at {GRAPH_PATH}")
class LLMSmokeTestCase(unittest.TestCase):
    @property
    def tokenizer(self):
        return _state["tokenizer"]

    @property
    def model(self):
        return _state["model"]

    @property
    def tool_fns(self):
        return _state["tool_fns"]


class TestInteractorCountBatchReliability(LLMSmokeTestCase):
    """Bug (the biggest single debugging saga this session): asking the
    model to make more than ~3-6 simultaneous count_nearby_entities calls
    in one turn pushed its greedy decoding into generating garbled
    <tool_call> tags (a literal stray "Ronaldo" token observed in place of
    the tag), silently dropping calls -- and the model's NEXT turn
    sometimes fabricated plausible-looking results for exactly the missing
    ones. Fixed by capping simultaneous batches at 3 (INTERACTOR_COUNT_NUDGE_BATCH)
    and a structural fabrication scrubber as a backstop. This test doesn't
    control the model's behavior directly (that's the whole point -- it's
    checking emergent behavior, not a deterministic function) but asserts
    the OUTCOME the fixes are supposed to guarantee: no fabrication
    artifacts reach the final answer, regardless of what the model does
    internally."""

    def test_compound_interactor_query_has_no_fabrication_markers(self):
        from query_interface import answer_query
        query = ("report the 3 people with maniest unique interactions. for each of them, report all "
                 "of the people they interacted with. for each interactor, report how many interaction "
                 "he has.")
        answer = answer_query(self.tokenizer, self.model, self.tool_fns, query,
                               max_iters=60, verbose=False)
        # A literal <tool_response> tag in the final answer is proof the model fabricated fake tool
        # output itself -- that specific string should never survive to the user (see
        # _strip_fabricated_count_claims and the <tool_response> guard in answer_query).
        self.assertNotIn("<tool_response>", answer)
        # The known garbling artifact itself should never appear in a final answer.
        self.assertNotIn("Ronaldo", answer)

    def test_compound_interactor_query_reaches_a_real_answer(self):
        from query_interface import answer_query
        query = "rank the top 3 people by unique interactions, then detail each one's interactions."
        answer = answer_query(self.tokenizer, self.model, self.tool_fns, query,
                               max_iters=30, verbose=False)
        self.assertNotIn("[stopped: max iterations reached]", answer)
        self.assertNotIn("[note: reached max tool-call iterations]", answer)
        # Should contain real deterministic-report content, not just free text -- e.g. the literal
        # "confirmed interactions" phrasing that only ever comes from a real tool result.
        self.assertIn("confirmed interaction", answer.lower())


class TestInventedIdRecovery(LLMSmokeTestCase):
    """Bug: the model sometimes bundles a ranking call with dependent detail
    calls in the SAME turn (before it could have seen the ranking result)
    and invents placeholder ids to fill the gap -- observed concretely as
    12345, 67890, 54321. Confirms the live recovery nudge (see
    format_invented_id_recovery in test_regressions.py for the
    unit-tested formatting logic) actually results in a correct final
    answer, not just a correctly-formatted nudge message."""

    def test_query_does_not_end_in_a_fabricated_no_evidence_claim(self):
        from query_interface import answer_query
        query = "report the 3 people with maniest unique interactions and detail each one's interactions."
        answer = answer_query(self.tokenizer, self.model, self.tool_fns, query,
                               max_iters=30, verbose=False)
        # The exact failure this bug produced: giving up with "not enough evidence" about invented
        # ids, while ignoring the real ranking result that was sitting in the same tool-call batch.
        lowered = answer.lower()
        gave_up_on_invented_ids = "12345" in answer or "67890" in answer or "54321" in answer
        self.assertFalse(gave_up_on_invented_ids and "not enough evidence" in lowered,
                          "answer appears to have given up on invented placeholder ids instead of recovering")


if __name__ == "__main__":
    unittest.main()
