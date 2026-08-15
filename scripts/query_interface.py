"""
Natural-language query interface over the event graph. An LLM
(Qwen2.5-7B-Instruct, chosen for reliable native tool-calling in its
chat template -- verified directly rather than assumed) is given the
GraphTools functions as callable tools and iterates: call a tool, read
the result, call another tool if needed, until it can answer.

System prompt explicitly instructs the model to answer ONLY from tool
results and to say so when evidence is thin -- the same "report only
high-confidence matches, and explicit I-don't-know" pattern discussed
earlier for this project's RAG design, now applied to a real
implementation instead of just a plan.

Usage:
  python scripts/query_interface.py --graph data/scene_061/event_graph_with_attrs.gpickle \
      "Which entities appeared in more than one camera?"
"""
import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from graph_tools import GraphTools

# Below this, an entity's caption disagreed across its sampled crops enough to be treated as
# contested rather than trusted (see extract_entity_attributes.py's caption_agreement field).
CAPTION_AGREEMENT_THRESHOLD = 0.6

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "search_by_appearance",
            "description": "Find ALL entities whose visual appearance description matches a text query (e.g. 'a man in "
                            "a grey shirt') at or above min_similarity -- NOT just the single best match. The tracking "
                            "pipeline behind this graph fragments each real person into many separate entities "
                            "(measured ~45x on average), so a description almost always corresponds to SEVERAL "
                            "different entities, not one -- even if one of them has a suspiciously perfect 1.0 "
                            "similarity (e.g. because its caption happens to repeat your query text verbatim), that "
                            "does not mean it's the only or even the correct match. Always use the default "
                            "min_similarity and look at the whole returned list, not just the top result. Each "
                            "result also has caption_agreement (0-1, from captioning the entity's crops several "
                            "times independently) -- below 0.6 means the crops described this entity inconsistently, "
                            "so treat the caption itself as contested, separate from how well it matched your query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "min_similarity": {"type": "number", "default": 0.7,
                                        "description": "Only return matches at or above this similarity. Do not "
                                                        "lower this just because the top match looks perfect -- "
                                                        "lower it only if zero results come back and you need to "
                                                        "broaden the search."},
                    "top_k": {"type": "integer", "default": 20, "description": "Safety cap on result count."},
                },
                "required": ["description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_entity_timeline",
            "description": "Get the full sighting history (which cameras, when, where) for one entity by its global_id.",
            "parameters": {
                "type": "object",
                "properties": {"global_id": {"type": "integer"}},
                "required": ["global_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_entities_in_camera",
            "description": "List entities seen in a given camera, optionally within a time window (seconds).",
            "parameters": {
                "type": "object",
                "properties": {
                    "camera_id": {"type": "string"},
                    "start_time_sec": {"type": "number"},
                    "end_time_sec": {"type": "number"},
                },
                "required": ["camera_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_entities_cooccur",
            "description": "Check whether two entities (by global_id) were ever seen in the same camera at overlapping times.",
            "parameters": {
                "type": "object",
                "properties": {
                    "global_id_a": {"type": "integer"},
                    "global_id_b": {"type": "integer"},
                    "max_gap_sec": {"type": "number", "default": 5.0},
                },
                "required": ["global_id_a", "global_id_b"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_nearby_entities",
            "description": "Find entities whose real-world trajectory actually came within a distance threshold "
                            "(meters) of the given entity's, in the same camera at matching timestamps. ONLY use "
                            "this when you already have a specific known global_id (e.g. from a previous tool "
                            "result) -- if the entity is described in words (e.g. 'the woman in a red dress'), use "
                            "find_nearby_entities_by_description instead, NOT this plus a separate "
                            "search_by_appearance call. check_entities_cooccur only tells you two entities were in "
                            "the same camera around the same time, NOT that they were physically near each other. "
                            "Returns confirmed_proximity_matches: real, trustworthy proximity events -- fine to "
                            "call these 'confirmed', but that word covers the DISTANCE measurement only, not "
                            "whether the matched entity's own caption is accurate. (Tentative and likely-same-"
                            "person matches are computed internally but deliberately excluded from this report.) "
                            "If asked how many interactions EACH returned match/interactor has of their own, call "
                            "count_nearby_entities (NOT this tool again) with that interactor's global_id -- it "
                            "does not come back precomputed.",
            "parameters": {
                "type": "object",
                "properties": {
                    "global_id": {"type": "integer"},
                    "max_distance_m": {
                        "type": "number", "default": 2.0,
                        "description": "Distance threshold IN METERS -- convert units if the user gives cm/ft "
                                        "(e.g. 'within 69cm' means max_distance_m=0.69). If the query gives no "
                                        "explicit threshold, leave this at the default.",
                    },
                    "max_gap_sec": {"type": "number", "default": 1.0},
                },
                "required": ["global_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_nearby_entities_by_description",
            "description": "For 'close to' / 'passed by' / 'interacted with' questions about an entity described "
                            "IN WORDS (e.g. 'a woman in a red dress'). Use this instead of manually chaining "
                            "search_by_appearance + find_nearby_entities -- it internally resolves the description "
                            "to ALL matching candidates (given known ~45x per-person track fragmentation, there "
                            "are usually several) and checks proximity for EVERY one of them, not just the best "
                            "appearance match, then merges the results. Returns candidates_checked (which "
                            "global_ids were actually checked) plus confirmed_proximity_matches, same as "
                            "find_nearby_entities. If asked how many interactions EACH returned match has of "
                            "their own, call count_nearby_entities (NOT find_nearby_entities again) with that "
                            "match's global_id -- it does not come back precomputed. "
                            "IMPORTANT: each match has TWO different people's info in it -- "
                            "global_id/appearance_caption describe the OTHER, nearby person; "
                            "searched_entity_global_id/searched_entity_caption describe the person who matched "
                            "your text description. Do not swap them (e.g. don't attribute searched_entity_"
                            "caption's activity/location claims to the other person).",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {"type": "string"},
                    "max_distance_m": {
                        "type": "number", "default": 2.0,
                        "description": "Distance threshold IN METERS -- convert units if the user gives cm/ft "
                                        "(e.g. 'within 69cm' means max_distance_m=0.69).",
                    },
                    "max_gap_sec": {"type": "number", "default": 1.0},
                },
                "required": ["description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "count_nearby_entities",
            "description": "Just the confirmed-interaction COUNT for one entity (an integer), not the full list "
                            "of who/where/when. Use this instead of find_nearby_entities whenever the question "
                            "only needs a NUMBER -- e.g. after find_nearby_entities or "
                            "find_nearby_entities_by_description returns several matches and you're asked how "
                            "many interactions each of THOSE matches has of their own. Calling find_nearby_entities "
                            "again per match for this would fetch far more detail than needed and produce an "
                            "unreadably long answer -- call this instead, once per match.",
            "parameters": {
                "type": "object",
                "properties": {
                    "global_id": {"type": "integer"},
                    "max_distance_m": {"type": "number", "default": 2.0},
                    "max_gap_sec": {"type": "number", "default": 1.0},
                },
                "required": ["global_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "list_multi_camera_entities",
            "description": "List all entities confirmed to have appeared in more than one camera.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "rank_entities_by_interaction_count",
            "description": "For 'who has the most interactions' / 'busiest people' / 'top N by encounters' style "
                            "questions. Computes, for EVERY entity in the graph, how many confirmed (not "
                            "uncertain, not likely-same-person) proximity matches it has, and returns the top_k "
                            "with the highest count. Use this instead of calling find_nearby_entities yourself "
                            "once per entity -- with hundreds of entities that is both infeasible within your "
                            "tool-call budget and, even if it weren't, counting/ranking that many results in text "
                            "is unreliable. This call itself takes a couple of minutes since it checks every "
                            "entity -- that's expected, not a hang.",
            "parameters": {
                "type": "object",
                "properties": {
                    "max_distance_m": {"type": "number", "default": 2.0,
                                        "description": "Distance threshold IN METERS for what counts as an interaction."},
                    "max_gap_sec": {"type": "number", "default": 1.0},
                    "top_k": {"type": "integer", "default": 10, "description": "How many top entities to return."},
                },
                "required": [],
            },
        },
    },
]

SYSTEM_PROMPT = """You are a multi-camera surveillance query assistant. You answer questions \
about people tracked across several cameras by calling the provided tools to look up real data \
-- never guess or invent camera names, times, or descriptions.

Rules:
- CHECK SCOPE FIRST: every tool here answers a specific, narrow question about a NAMED target -- a \
particular entity (by description or global_id), a particular camera, or a particular time window. None \
of them define or detect abstract qualities like "suspicious," "unusual," "strange," or "notable" -- \
there is no tool that scores or flags behavior as anomalous. If the query does not name or describe a \
specific entity/camera/time to investigate (e.g. "detect suspicious behavior," "find anything unusual"), \
do NOT invent a target by grabbing an arbitrary entry from an unrelated prior tool result (e.g. the first \
item in a list_multi_camera_entities result) and investigating that as if it were what was asked. \
Instead, say directly that the available tools only support targeted lookups (by description, global_id, \
camera, or time) and cannot evaluate an undefined quality like "suspicious" -- then, if possible, ask what \
specific pattern would count (e.g. "many different people converging in one place," "unusually long \
encounters") so it could be checked with the tools that do exist.
- Answer ONLY using information returned by tool calls. Do not use outside knowledge or assumptions.
- If the query describes an entity in words (e.g. "the woman in a red dress") rather than by global_id, \
you MUST call search_by_appearance FIRST to resolve it to a real global_id before calling any other \
tool on it. Never invent or guess a global_id -- every global_id you use in a tool call or in your \
answer must have come from a previous tool result, either from search_by_appearance or from another \
tool's output (e.g. a nearby-entity result).
- For "close to" / "passed by" style questions about a WORDED description, use \
find_nearby_entities_by_description directly -- do NOT manually call search_by_appearance followed by a \
single find_nearby_entities call, that only checks one candidate and silently ignores the rest.
- search_by_appearance returns MULTIPLE candidate entities on purpose -- treat all of them as possibly \
relevant, not just the top one. A 1.0-similarity result is not more "correct" than the others, it just \
means that entity's caption happens to repeat your query text -- it is still only one fragment among many.
- Captions can be confidently wrong, especially about ACTIVITY or SCENE context (e.g. captions mentioning \
"playing tennis" in this dataset are a known hallucination -- there is no tennis in these videos; the \
model free-associates from a court-like floor texture). Color/clothing-type words in a caption are more \
trustworthy than claims about what someone is doing or where. When a caption's activity claim seems \
suspicious or irrelevant to the question, ignore that part rather than repeating it as fact.
- Every entity's caption was independently generated from several different crops of it; caption_agreement \
(0-1) measures how much those crops agreed. This dataset measured caption_agreement < 0.6 for roughly 3 \
out of 4 entities -- so a low or missing caption_agreement is common, not rare, and should make you MORE \
cautious about repeating that caption as fact, not just a minor footnote.
- Cite your evidence: mention camera ids and timestamps that support your answer. Report time using the \
tool's "start_time"/"end_time"/"time" fields (M:SS format) rather than the raw "*_time_sec" fields -- do \
not convert seconds to M:SS yourself, use the pre-computed field so you can't get the arithmetic wrong.
- If the tool results don't give you enough evidence to answer confidently, say so explicitly \
instead of guessing -- e.g. "I don't have enough evidence in the tracked data to confirm this."
- The underlying tracking/detection is imperfect (this is a known limitation of the pipeline), so \
prefer conservative, hedged answers over confident-sounding but unsupported ones.
- "Close to" / "passed by" / "near" questions are about physical distance, not just being in the \
same camera around the same time -- use find_nearby_entities (known global_id) or \
find_nearby_entities_by_description (worded description) for those, not check_entities_cooccur.
- find_nearby_entities and find_nearby_entities_by_description return confirmed_proximity_matches: real \
proximity events only. Fine to call these "confirmed" -- but that word covers the DISTANCE measurement \
only, not whether the matched entity's own caption (e.g. an activity claim) is accurate -- keep those two \
kinds of confidence separate in what you say.
- If confirmed_proximity_matches is empty, say so plainly ("no confirmed proximity matches found") rather \
than guessing or inventing an encounter.
- find_nearby_entities, find_nearby_entities_by_description, and rank_entities_by_interaction_count all \
return pre-formatted, already-correct text (not raw JSON) as their tool result -- every fact in that text \
is guaranteed accurate and is automatically appended to your final answer verbatim, in the order you \
called them, no matter how many of these calls you make. Because of this, you do NOT need to (and \
should NOT) retype, paraphrase, or summarize the facts from these reports in your own final answer -- \
write only a short framing/narration sentence or two; the detailed data will appear right after it \
automatically. This also means you should feel free to call these tools MULTIPLE times in sequence for \
compound queries -- e.g. "rank the top 3 by interactions, then detail each one's interactions" needs one \
ranking call FOLLOWED BY one detail call per entity in that ranking, not just the first call. Keep going \
until you've covered every part of what the query actually asked before stopping.
- Planning several tool calls at once tends to pull you into explaining the plan in prose (e.g. a numbered \
list saying "I need to call X for each of these") instead of actually calling them. Explaining a call is \
NOT the same as making one -- if you notice yourself about to write a sentence like "I need to call X" or \
"I will now call X", stop and emit the real <tool_call>{"name": ..., "arguments": ...}</tool_call> block(s) \
instead of describing them. You can emit multiple <tool_call> blocks in the same turn ONLY when you \
already know every argument for all of them from a PRIOR tool result -- e.g. calling find_nearby_entities \
once per id you already have. Do NOT bundle a ranking/search call together with follow-up detail calls in \
the same turn -- you cannot know which ids to detail until you actually see that call's result, and \
inventing placeholder ids to fill the gap (e.g. guessing round numbers like 12345) is exactly the kind of \
invented global_id the rule above forbids. Call the ranking/search tool alone first, wait for its real \
result, THEN call the detail tools with the real ids it returned.
"""


def agreement_flag(agreement) -> str:
    # Multiple crops of the same entity were captioned independently; low agreement means they
    # described it very differently -- a caption-level hallucination risk, distinct from (and on
    # top of) any tracking/distance-based confidence issue.
    if agreement is not None and agreement < CAPTION_AGREEMENT_THRESHOLD:
        return f" [caption uncertain, agreement={agreement}]"
    return ""


def format_ranking_report(result: list) -> str:
    """Deterministically render a rank_entities_by_interaction_count result. No LLM involved -- the
    count itself was computed once in Python (len() on each entity's confirmed_proximity_matches),
    not derived by the model reading/counting a list in its own context."""
    if not result:
        return "No entities had any confirmed proximity matches within the given thresholds."
    lines = [f"Top {len(result)} entities by confirmed interaction count:"]
    for r in result:
        flag = agreement_flag(r.get("caption_agreement"))
        lines.append(f"  - global_id {r['global_id']}: {r['confirmed_interaction_count']} confirmed "
                      f"interactions -- {r['appearance_caption']}{flag}")
    ids = ", ".join(str(r["global_id"]) for r in result)
    lines.append(f"\nThis list does NOT say who any of these entities interacted with -- only their counts. "
                 f"If the question asks for that detail, you MUST call find_nearby_entities separately for EACH "
                 f"of these exact global_ids before answering: {ids}. Do not describe, guess, or reuse another "
                 f"entity's interactions for any global_id you have not called find_nearby_entities on yourself. "
                 f"Do this now, in your very next turn, by emitting real <tool_call>{{\"name\": "
                 f"\"find_nearby_entities\", \"arguments\": {{...}}}}</tool_call> blocks -- one per global_id, "
                 f"all in the same turn if you can. Writing a sentence like \"I need to call find_nearby_entities "
                 f"for global_id 7\" does NOT call it; only an actual <tool_call> block does.")
    return "\n".join(lines)


# Tools whose results are reported via a deterministic template instead of LLM free text. Repeated
# testing this session showed the LLM reliably corrupting facts (captions, distances, ids) even when
# the retrieved data was already correct -- e.g. copying one match's caption onto an adjacent, different
# match, or miscounting a list it was just given verbatim. Every fix at the data/tool layer held up
# under verification; every fix attempted purely via prompt wording did not. So for these tools, the
# actual facts never pass through free-text generation at all -- code formats them directly from the
# tool's JSON, verbatim.
DETERMINISTIC_REPORT_FORMATTERS = {
    "find_nearby_entities": lambda result: format_proximity_report(result),
    "find_nearby_entities_by_description": lambda result: format_proximity_report(result),
    "rank_entities_by_interaction_count": lambda result: format_ranking_report(result),
    "count_nearby_entities": lambda result: format_count_report(result),
}
DETERMINISTIC_REPORT_TOOLS = set(DETERMINISTIC_REPORT_FORMATTERS)


def format_count_report(result: dict) -> str:
    """Deterministically render a count_nearby_entities result -- a single fact, but still routed
    through this instead of LLM free text for the same reason as every other deterministic report."""
    if "error" in result:
        return f"Error: {result['error']}"
    flag = agreement_flag(result.get("caption_agreement"))
    return (f"global_id {result['global_id']}: {result['confirmed_interaction_count']} confirmed "
            f"interactions of their own -- {result.get('appearance_caption')}{flag}")


INTERACTOR_COUNT_NUDGE_BATCH = 3  # ids nudged per iteration -- tested empirically at both 15 and 6:
# batches of that size push this model's greedy decoding into generating garbled <tool_call> tags
# partway through (a literal stray "Ronaldo" token in place of the tag), silently dropping calls, and
# the model then fabricates plausible-looking counts for exactly the missing ones on its next turn --
# sometimes wrapped in a fake <tool_response> tag, sometimes as bare prose matching the real report
# template closely enough to evade a tag-only check (see _strip_fabricated_count_claims, the structural
# backstop for whatever gets through). 3 simultaneous tool calls per turn was reliable throughout this
# whole session (e.g. the rank -> per-entity detail step), so that's the batch size -- NOT a cap on
# total coverage. Every discovered interactor is queued and nudged a batch at a time, across as many
# iterations as it takes, until the queue is drained or max_iters runs out (see answer_query).


def format_proximity_report(result: dict) -> str:
    """Deterministically render a find_nearby_entities / find_nearby_entities_by_description result.
    No LLM involved -- every fact here is copied verbatim from the tool's own JSON output. Only
    confirmed_proximity_matches is shown -- uncertain_proximity_matches (short-track-fragment, tentative)
    and likely_same_person_matches (not real interactions at all) are still computed by the underlying
    tool and available in its raw result, but are deliberately left out of this report on request.
    Does NOT include an interactor-count nudge -- see format_interactor_count_nudge, invoked once per
    iteration (not once per call) by answer_query, for why that has to be handled separately."""
    if "error" in result:
        return f"Error: {result['error']}"

    confirmed = result.get("confirmed_proximity_matches", [])

    def fmt_match(m):
        flag = agreement_flag(m.get("caption_agreement"))
        return (f"  - camera {m['camera']}, {m['time']}: {m['appearance_caption']}{flag} "
                f"(global_id {m['global_id']}), {m['distance_m']}m away")

    lines = []
    if confirmed:
        lines.append(f"Confirmed proximity matches ({len(confirmed)}):")
        lines += [fmt_match(m) for m in confirmed]
    else:
        lines.append("No confirmed proximity matches found within the given distance/time thresholds.")

    return "\n".join(lines)


def format_proximity_report_merged(result: dict, count_results: dict) -> str:
    """Same as format_proximity_report, but for the FINAL answer shown to the user only: inlines each
    match's own interaction count on the same line when a count_nearby_entities call for that
    global_id has already completed, instead of leaving it as a separate block the reader has to
    cross-reference by id themselves. Only used for final display -- the model still sees the plain,
    un-merged per-call reports live during the loop (see answer_query), since those are already correct
    and re-deriving them mid-conversation isn't necessary."""
    if "error" in result:
        return f"Error: {result['error']}"

    confirmed = result.get("confirmed_proximity_matches", [])

    def fmt_match(m):
        flag = agreement_flag(m.get("caption_agreement"))
        count_result = count_results.get(m["global_id"])
        count_str = ""
        if count_result and "confirmed_interaction_count" in count_result:
            count_str = f", {count_result['confirmed_interaction_count']} confirmed interactions of their own"
        return (f"  - camera {m['camera']}, {m['time']}: {m['appearance_caption']}{flag} "
                f"(global_id {m['global_id']}), {m['distance_m']}m away{count_str}")

    lines = []
    if confirmed:
        lines.append(f"Confirmed proximity matches ({len(confirmed)}):")
        lines += [fmt_match(m) for m in confirmed]
    else:
        lines.append("No confirmed proximity matches found within the given distance/time thresholds.")

    return "\n".join(lines)


def format_deterministic_summary(proximity_calls: dict) -> str:
    """A short, code-generated summary line per entity that was investigated via find_nearby_entities /
    find_nearby_entities_by_description this query -- placed right after the model's own framing text.
    Exists because the model's free-text framing was observed (via debug trace) to sometimes duplicate
    one entity's interactor list under a DIFFERENT entity's heading -- correct data, wrong attribution.
    This doesn't replace the model's framing (still useful as a natural-language lead-in) -- it adds a
    guaranteed-correct quick-reference right beneath it, so a misattributed sentence above it can't be
    the reader's only source for these specific facts."""
    if not proximity_calls:
        return ""
    lines = ["Summary (guaranteed accurate -- generated directly from tool results, not the model):"]
    for (name, key), (header, result) in proximity_calls.items():
        if "error" in result:
            continue
        gid = key if isinstance(key, int) else result.get("target_global_id")
        caption = result.get("target_caption") or ""
        count = len(result.get("confirmed_proximity_matches", []))
        label = f"global_id {gid}" if gid is not None else f"description {key!r}"
        lines.append(f"  - {label}: {count} confirmed interactions -- {caption}".rstrip(" -"))
    return "\n".join(lines)


def format_interactor_count_nudge(batch_ids: list, remaining_after: int) -> str:
    """A single, consolidated instruction naming the NEXT batch of global_ids to call
    count_nearby_entities on. Deliberately a flat list, not grouped by source entity -- an earlier,
    grouped version caused the model to conflate several simultaneous groups (see git history). This
    is called repeatedly, once per iteration, draining a work queue a few ids at a time (see
    INTERACTOR_COUNT_NUDGE_BATCH's docstring for why the batch itself stays small) until every
    known interactor has been asked about -- remaining_after tells the model more batches are coming
    so it doesn't treat this one batch as the whole remaining task and try to answer prematurely."""
    if not batch_ids:
        return ""
    ids = ", ".join(str(i) for i in batch_ids)
    more = (f" There are {remaining_after} more after this batch -- you'll be prompted for those in "
            f"following turns, do not try to guess or skip ahead to them now." if remaining_after > 0 else "")
    return (f"\nNone of the reports above say how many interactions of their OWN each matched entity has -- "
            f"only their distance from the entity you searched for. If the question asks for each "
            f"interactor's own count, you MUST call count_nearby_entities separately for EACH of these exact "
            f"global_ids before answering: {ids}. Do not describe, guess, or reuse another entity's "
            f"interaction count for any global_id you have not called count_nearby_entities on "
            f"yourself.{more} Do this now, in your very next turn, by emitting real <tool_call>{{\"name\": "
            f"\"count_nearby_entities\", \"arguments\": {{...}}}}</tool_call> blocks -- one per global_id, all "
            f"in the same turn if you can. Writing a sentence describing the call does NOT call it; only an "
            f"actual <tool_call> block does. If the question does NOT ask for interactor-level counts, ignore "
            f"this and answer normally.")


def _write_debug_dump(trace: dict, debug_dir: Path) -> Path:
    """Writes the full trace of one query -- every rendered prompt actually sent to the model, every
    tool call and its raw JSON result, and how the final answer was produced -- to its own JSON file.
    Separate from --verbose (which prints a live but ephemeral console trace): this is a persistent,
    complete, machine-readable dump meant for after-the-fact inspection of exactly what data the
    answer was built from."""
    debug_dir.mkdir(parents=True, exist_ok=True)
    slug = "".join(c if c.isalnum() else "_" for c in trace["query"])[:40]
    path = debug_dir / f"{trace['timestamp']}_{slug}.json"
    with open(path, "w") as f:
        f.write(_json_dumps_compact_lists(trace))
    return path


def _json_dumps_compact_lists(obj, level: int = 0, indent: int = 2) -> str:
    """Like json.dumps(obj, indent=2), except a list containing only flat primitives (e.g. a
    token-id list with thousands of ints) is printed on ONE line instead of one element per line --
    indent=2 alone makes those unreadable. Lists of dicts (iterations, tool_calls, ...) still get
    normal multi-line indentation."""
    pad, pad_in = " " * (level * indent), " " * ((level + 1) * indent)
    if isinstance(obj, dict):
        if not obj:
            return "{}"
        items = [f'{pad_in}{json.dumps(k)}: {_json_dumps_compact_lists(v, level + 1, indent)}' for k, v in obj.items()]
        return "{\n" + ",\n".join(items) + "\n" + pad + "}"
    if isinstance(obj, list):
        if not obj:
            return "[]"
        if all(isinstance(x, (int, float, str, bool)) or x is None for x in obj):
            return json.dumps(obj, default=str)
        items = [f'{pad_in}{_json_dumps_compact_lists(x, level + 1, indent)}' for x in obj]
        return "[\n" + ",\n".join(items) + "\n" + pad + "]"
    return json.dumps(obj, default=str)


_COUNT_CLAIM_PATTERN = re.compile(r"global_id (\d+): \d+ confirmed interactions? of (their|his|her|its) own[^\n]*")


def _strip_fabricated_count_claims(intro: str, deterministic_reports: list) -> str:
    """Defense against a real, observed failure mode: when a large batch of count_nearby_entities
    calls partially fails to parse (see INTERACTOR_COUNT_NUDGE_BATCH's docstring -- greedy decoding
    can garble <tool_call> tags mid-batch), the model's NEXT turn sometimes invents a plausible-looking
    count for exactly the ids that never actually got a real result -- as bare prose matching the
    deterministic-report template, not always wrapped in a detectable <tool_response> tag. A tag check
    alone missed this. This instead cross-checks every "global_id N: ... confirmed interactions" claim
    in the model's own intro text against the ids that ACTUALLY have a real deterministic report --
    format alone can't fake its way past this, since it's checked against verified data, not a pattern."""
    verified_ids = {int(m) for m in re.findall(r"global_id (\d+): \d+ confirmed interactions? of "
                                                 r"(?:their|his|her|its) own", "\n".join(deterministic_reports))}

    def _scrub(match):
        gid = int(match.group(1))
        if gid in verified_ids:
            return match.group(0)
        return f"[claim about global_id {gid}'s interaction count removed -- no verified tool result for it]"

    return _COUNT_CLAIM_PATTERN.sub(_scrub, intro)


def _combine_answer(intro: str, deterministic_reports: list, display_reports: list = None) -> tuple:
    """Combine the model's own free text with every deterministic report gathered during the whole
    investigation, however many tool calls that took. The model's text is narration/framing only --
    every actual fact (caption, distance, count, id) comes from the accumulated reports, appended
    verbatim, in call order, never retyped by the model. This is what lets a compound query ("rank
    the top 3, then detail each") stay fact-safe across many tool calls instead of only the single
    narrow case a hardcoded short-circuit could recognize.
    display_reports, if given, is what's actually shown to the user (e.g. proximity matches with each
    interactor's own count inlined -- see format_proximity_report_merged) while deterministic_reports
    (the plain, un-merged per-call reports) is still what the fabrication check verifies against --
    the two can differ in FORMATTING without affecting which facts are considered verified."""
    intro = _strip_fabricated_count_claims(intro, deterministic_reports)
    reports = display_reports if display_reports is not None else deterministic_reports
    if not reports:
        return intro, "llm_free_text"
    return intro.strip() + "\n\n" + "\n\n".join(reports), "llm_framing_plus_deterministic_reports"


def _mentions_unexecuted_tool(text: str, tool_names) -> str:
    """Returns the first real tool name found in text, or None. Only meaningful when called on a
    response that _extract_tool_calls already found zero real <tool_call> blocks in -- a tool's exact
    function name (e.g. "find_nearby_entities") essentially never appears in ordinary prose otherwise,
    so finding one here is a reliable, cheap signal that the model narrated an intention to call it
    ("I need to call find_nearby_entities(global_id=7)...") without emitting the syntax that would
    actually do so. Caught directly: a compound query where the model correctly named all 3 required
    calls in words, then stopped without calling any of them."""
    for name in tool_names:
        if name in text:
            return name
    return None


def answer_query(tokenizer, model, tool_fns, query: str, max_iters: int, verbose: bool,
                  debug: bool = False, debug_dir: Path = None) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    trace = {"query": query, "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S_%f"),
              "max_iters": max_iters, "iterations": []} if debug else None
    deterministic_reports = []  # accumulated across every deterministic-report tool call this turn --
    # fed to the model live, per call, AND used as the fabrication check's source of verified facts
    nudges_remaining = 2  # bounded retries if the model narrates a tool call instead of making it

    pending_count_ids = []  # FIFO queue of interactor global_ids discovered (via find_nearby_entities /
    # find_nearby_entities_by_description) but not yet passed to count_nearby_entities -- drained
    # INTERACTOR_COUNT_NUDGE_BATCH ids at a time, once per iteration, until empty or max_iters runs out.
    # A one-shot nudge only ever covered the first few interactors an entity had; this keeps going
    # across as many iterations as it takes to cover every one that was discovered.
    seen_interactor_ids = set()  # every id ever added to pending_count_ids or count_results -- dedup

    # Tracked separately from deterministic_reports purely for FINAL DISPLAY formatting -- merging each
    # interactor's own count into its proximity-match line instead of showing it as a disconnected
    # count_nearby_entities(...) block the reader has to cross-reference by id (see
    # format_proximity_report_merged). Keyed to dedupe repeated identical calls automatically.
    proximity_calls = {}  # (name, gid-or-description) -> (header, result)
    rank_report_texts = []  # formatted rank_entities_by_interaction_count reports, in call order
    count_results = {}  # global_id -> count_nearby_entities result

    known_entity_ids = set()  # every global_id CONFIRMED to exist via a real tool result so far --
    # used to recover from a real observed failure: the model sometimes bundles a ranking/search call
    # together with follow-up detail calls in the SAME turn, before it can have seen the ranking
    # result, and invents placeholder ids (e.g. 12345) to fill the gap. Those calls correctly error out
    # ("no entity with global_id=...") at the tool layer, but the model then sometimes gives up ("not
    # enough evidence") instead of noticing the REAL result was sitting right there in the same batch.

    def build_display_reports() -> list:
        summary = format_deterministic_summary(proximity_calls)
        reports = ([summary] if summary else []) + rank_report_texts + [
            f"{header}\n{format_proximity_report_merged(result, count_results)}"
            for header, result in proximity_calls.values()
        ]
        return reports

    def finish(answer: str, source: str) -> str:
        if trace is not None:
            trace["final_answer"] = answer
            trace["answer_source"] = source
            path = _write_debug_dump(trace, debug_dir)
            print(f"[debug dump written to {path}]")
        return answer

    for iteration in range(max_iters):
        prompt = tokenizer.apply_chat_template(
            messages, tools=TOOL_SCHEMAS, add_generation_prompt=True, tokenize=False
        )
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        new_token_ids = out_ids[0][inputs.input_ids.shape[1]:]
        response_text = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()

        if verbose:
            print(f"\n--- iteration {iteration} raw output ---\n{response_text}\n")

        # The two stages between rendered_prompt (text in) and raw_model_output (text out): the actual
        # token ids on both sides of generate(), with nothing decoded/interpreted yet -- e.g. this is
        # what would show a special/control token that skip_special_tokens quietly dropped from the text.
        iter_trace = {
            "iteration": iteration, "rendered_prompt": prompt,
            "input_token_ids": inputs.input_ids[0].tolist(), "input_token_count": int(inputs.input_ids.shape[1]),
            "output_token_ids": new_token_ids.tolist(), "output_token_count": int(new_token_ids.shape[0]),
            "raw_model_output": response_text, "tool_calls": [],
        } if trace is not None else None

        tool_calls = _extract_tool_calls(response_text)

        # Real tool results only ever enter the conversation as a {"role": "tool"} message WE inject --
        # the model itself never legitimately produces "<tool_response>" markup. Seeing it in the
        # model's own generation is a reliable fabrication signature: observed concretely when a large
        # tool-call batch degenerated mid-generation (garbled <tool_call> tags caused several calls to
        # silently fail to parse), and the model's NEXT turn simply invented plausible-looking fake
        # <tool_response> blocks with made-up numbers to paper over the gap, formatted identically to
        # (and thus indistinguishable from) genuine deterministic reports. Never let this reach the user.
        if "<tool_response>" in response_text and nudges_remaining > 0:
            nudges_remaining -= 1
            if verbose:
                print(f"\n--- fabricated <tool_response> markup detected; rejecting and nudging "
                      f"({nudges_remaining} retries left) ---\n")
            if iter_trace is not None:
                trace["iterations"].append(iter_trace)
            messages.append({"role": "assistant", "content": response_text})
            messages.append({"role": "user", "content":
                "You generated fake <tool_response> content yourself -- real tool results only ever come "
                "from an actual tool call you make, never from you writing them directly. Some of your "
                "previous tool calls may not have gone through; do not guess or invent their results. "
                "Re-emit real <tool_call>{\"name\": ..., \"arguments\": ...}</tool_call> blocks for "
                "anything you still need, one at a time if a large batch failed before."})
            continue

        if not tool_calls:
            unexecuted = _mentions_unexecuted_tool(response_text, tool_fns.keys())
            if unexecuted and nudges_remaining > 0:
                nudges_remaining -= 1
                if verbose:
                    print(f"\n--- narrated but did not call {unexecuted}; nudging to actually call it "
                          f"({nudges_remaining} retries left) ---\n")
                if iter_trace is not None:
                    trace["iterations"].append(iter_trace)
                messages.append({"role": "assistant", "content": response_text})
                messages.append({"role": "user", "content":
                    f"You described calling {unexecuted} but did not actually call it. To call a tool "
                    f"you MUST emit a real <tool_call>{{\"name\": ..., \"arguments\": ...}}</tool_call> "
                    f"block -- describing it in words does not call it. Call it now."})
                continue
            if pending_count_ids:
                # The model tried to answer before every discovered interactor was counted -- keep
                # going instead of accepting a premature answer. Bounded by max_iters like everything
                # else here, not by a separate retry budget, since this is legitimate remaining work,
                # not an error state.
                batch, pending_count_ids[:] = pending_count_ids[:INTERACTOR_COUNT_NUDGE_BATCH], pending_count_ids[INTERACTOR_COUNT_NUDGE_BATCH:]
                nudge = format_interactor_count_nudge(batch, len(pending_count_ids))
                if verbose:
                    print(f"\n--- draining pending interactor-count queue ({len(pending_count_ids)} left "
                          f"after this batch) ---\n{nudge}\n")
                if iter_trace is not None:
                    trace["iterations"].append(iter_trace)
                messages.append({"role": "assistant", "content": response_text})
                messages.append({"role": "user", "content": nudge})
                continue
            if iter_trace is not None:
                trace["iterations"].append(iter_trace)
            answer, source = _combine_answer(response_text, deterministic_reports, build_display_reports())
            return finish(answer, source)

        messages.append({"role": "assistant", "content": response_text})
        invented_id_errors = []  # global_ids used this iteration that errored as nonexistent
        for call in tool_calls:
            name, call_args = call["name"], call.get("arguments", {})
            if name not in tool_fns:
                result = {"error": f"unknown tool {name}"}
            else:
                try:
                    result = tool_fns[name](**call_args)
                except Exception as e:
                    result = {"error": str(e)}
            if verbose:
                print(f"  tool call: {name}({call_args}) -> {json.dumps(result)[:300]}")
            if iter_trace is not None:
                iter_trace["tool_calls"].append({"name": name, "arguments": call_args, "result": result})
            if isinstance(result, dict) and "error" in result and "no entity with global_id=" in result["error"]:
                invented_id_errors.append(call_args.get("global_id"))
            elif isinstance(result, dict) and "error" not in result:
                # Track every global_id CONFIRMED real by a successful result, regardless of which tool
                # -- see known_entity_ids' declaration for why (recovering from invented placeholder ids).
                if "global_id" in result:
                    known_entity_ids.add(result["global_id"])
                if "target_global_id" in result:
                    known_entity_ids.add(result["target_global_id"])
                for key in ("confirmed_proximity_matches", "uncertain_proximity_matches", "likely_same_person_matches"):
                    known_entity_ids.update(m["global_id"] for m in result.get(key, []))
            elif isinstance(result, list):
                known_entity_ids.update(r["global_id"] for r in result if "global_id" in r)
            if name in DETERMINISTIC_REPORT_TOOLS and not (isinstance(result, dict) and "error" in result):
                # ALWAYS accumulate + feed back the formatted report, and NEVER short-circuit here --
                # doing so based on "which tool" or "which iteration" was itself just another narrow,
                # brittle special case (a real bug: it truncated a compound query -- "rank the top 3,
                # THEN detail each" -- right after the ranking step, because ranking legitimately is a
                # valid first move that still needed more investigation afterward). The model is free to
                # call as many of these as the query needs; every fact it could cite is guaranteed
                # correct by construction because none of them are ever retyped by the model -- see
                # _combine_answer, which appends these verbatim to whatever the model writes at the end.
                report = DETERMINISTIC_REPORT_FORMATTERS[name](result)
                header = f"--- {name}({', '.join(f'{k}={v!r}' for k, v in call_args.items())}) ---"
                if name in ("find_nearby_entities", "find_nearby_entities_by_description"):
                    for m in result.get("confirmed_proximity_matches", []):
                        gid = m["global_id"]
                        if gid not in seen_interactor_ids:
                            seen_interactor_ids.add(gid)
                            pending_count_ids.append(gid)
                    # Dedup key uses the call's own args (global_id or description), not source_gid --
                    # find_nearby_entities_by_description has no single source_gid (it aggregates over
                    # multiple candidates), so keying on that would silently merge two different
                    # description searches into one.
                    dedup_key = call_args.get("global_id", call_args.get("description"))
                    proximity_calls[(name, dedup_key)] = (header, result)
                elif name == "rank_entities_by_interaction_count":
                    rank_report_texts.append(f"{header}\n{report}")
                elif name == "count_nearby_entities" and "global_id" in result:
                    count_results[result["global_id"]] = result
                    seen_interactor_ids.add(result["global_id"])
                    if result["global_id"] in pending_count_ids:
                        pending_count_ids.remove(result["global_id"])
                deterministic_reports.append(f"{header}\n{report}")
                if verbose:
                    print(f"\n--- deterministic report (fact-guaranteed, not LLM-generated) ---\n{report}\n")
                messages.append({"role": "tool", "name": name, "content": report})
            else:
                messages.append({"role": "tool", "name": name, "content": json.dumps(result)})
        if pending_count_ids:
            batch, pending_count_ids[:] = pending_count_ids[:INTERACTOR_COUNT_NUDGE_BATCH], pending_count_ids[INTERACTOR_COUNT_NUDGE_BATCH:]
            nudge = format_interactor_count_nudge(batch, len(pending_count_ids))
            if verbose:
                print(f"\n--- interactor-count nudge ({len(pending_count_ids)} left after this batch) ---\n{nudge}\n")
            messages.append({"role": "user", "content": nudge})
        if invented_id_errors and known_entity_ids:
            # Recovery for a real observed failure: the model bundled a ranking/search call together
            # with follow-up detail calls in the same turn, invented placeholder ids (e.g. 12345) for
            # the latter since it couldn't have known the real ones yet, got real "no entity" errors
            # back, and then gave up ("not enough evidence") despite the REAL result being right there
            # in the same batch. Point it at the real ids explicitly instead of leaving it to notice.
            bad = ", ".join(str(g) for g in invented_id_errors)
            good = ", ".join(str(g) for g in sorted(known_entity_ids))
            recovery = (f"\nglobal_id(s) {bad} do not exist in this dataset -- do not reuse or guess ids "
                        f"like these. The REAL entity ids confirmed by your own tool results so far are: "
                        f"{good}. Use ONLY these real ids for any further detail calls this query needs.")
            if verbose:
                print(f"\n--- invented-id recovery nudge ---\n{recovery}\n")
            messages.append({"role": "user", "content": recovery})
        if iter_trace is not None:
            trace["iterations"].append(iter_trace)

    forced = _force_final_answer(tokenizer, model, messages, verbose, trace)
    answer, source = _combine_answer(forced, deterministic_reports, build_display_reports())
    return finish(answer + "\n\n[note: reached max tool-call iterations]", source)


def _force_final_answer(tokenizer, model, messages: list, verbose: bool, trace: dict = None) -> str:
    """Called when max_iters is exhausted mid tool-call loop. Re-prompts WITHOUT tool schemas so the
    model can't emit another tool call -- it must synthesize a text answer from whatever evidence it
    already gathered. Without this, the loop's last message is a raw tool-result dict, which is not
    an answer (this was a real bug: query_interface used to return that dict verbatim)."""
    messages = messages + [{"role": "user", "content":
        "You must answer now using only the tool results already gathered above. Do not call any more tools."}]
    prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
    new_token_ids = out_ids[0][inputs.input_ids.shape[1]:]
    text = tokenizer.decode(new_token_ids, skip_special_tokens=True).strip()
    if verbose:
        print(f"\n--- forced final answer ---\n{text}\n")
    if trace is not None:
        trace["iterations"].append({
            "iteration": "forced_final", "rendered_prompt": prompt,
            "input_token_ids": inputs.input_ids[0].tolist(), "input_token_count": int(inputs.input_ids.shape[1]),
            "output_token_ids": new_token_ids.tolist(), "output_token_count": int(new_token_ids.shape[0]),
            "raw_model_output": text, "tool_calls": [],
        })
    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("query", type=str, nargs="?", default=None,
                         help="Ask one question and exit. Omit to start an interactive session instead.")
    parser.add_argument("--graph", type=str, default=str(PROJECT_ROOT / "data" / "scene_061" / "event_graph_with_attrs.gpickle"))
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max_iters", type=int, default=60,
                         help="Needs to be high enough to check EVERY search_by_appearance candidate "
                              "(often ~9) with a follow-up tool call, not just the top one, plus draining "
                              "the full per-interactor count queue (INTERACTOR_COUNT_NUDGE_BATCH ids per "
                              "iteration) for compound queries asking each interactor's own count -- an "
                              "entity can have dozens of interactors, so full coverage genuinely needs "
                              "many iterations; lower this for faster-but-partial answers instead.")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--debug", action="store_true",
                         help="Dump every rendered prompt, tool call, and raw tool result for each "
                              "query to a JSON file under --debug_dir.")
    parser.add_argument("--debug_dir", type=str, default=str(PROJECT_ROOT / "data" / "scene_061" / "query_debug"))
    args = parser.parse_args()
    debug_dir = Path(args.debug_dir)

    tools = GraphTools(args.graph)
    tool_fns = {
        "search_by_appearance": tools.search_by_appearance,
        "get_entity_timeline": tools.get_entity_timeline,
        "find_entities_in_camera": tools.find_entities_in_camera,
        "check_entities_cooccur": tools.check_entities_cooccur,
        "find_nearby_entities": tools.find_nearby_entities,
        "count_nearby_entities": tools.count_nearby_entities,
        "find_nearby_entities_by_description": tools.find_nearby_entities_by_description,
        "list_multi_camera_entities": tools.list_multi_camera_entities,
        "rank_entities_by_interaction_count": tools.rank_entities_by_interaction_count,
    }

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    if args.query is not None:
        answer = answer_query(tokenizer, model, tool_fns, args.query, args.max_iters, args.verbose,
                               args.debug, debug_dir)
        print(f"\n=== Answer ===\n{answer}")
        return

    print(f"\nLoaded graph: {len(tools.entities)} entities. Ask a question, or 'quit' to exit.")
    print("Each question is independent -- no conversation memory between turns.\n")
    while True:
        try:
            query = input("You: ").strip()
        except EOFError:
            break
        if not query:
            continue
        if query.lower() in ("quit", "exit"):
            break
        answer = answer_query(tokenizer, model, tool_fns, query, args.max_iters, args.verbose,
                               args.debug, debug_dir)
        print(f"\nAssistant: {answer}\n")


def _extract_tool_calls(text: str) -> list:
    """Qwen2.5's tool-call format: one or more <tool_call>{json}</tool_call> blocks."""
    calls = []
    marker = "<tool_call>"
    end_marker = "</tool_call>"
    pos = 0
    while marker in text[pos:]:
        start = text.index(marker, pos) + len(marker)
        end = text.index(end_marker, start) if end_marker in text[start:] else len(text)
        try:
            calls.append(json.loads(text[start:end].strip()))
        except json.JSONDecodeError:
            pass
        pos = end + len(end_marker)
    return calls


if __name__ == "__main__":
    main()
