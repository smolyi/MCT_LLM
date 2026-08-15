"""
Quantitative eval for the LLM query interface: a curated set of natural-
language queries, each with an expected answer computed DIRECTLY from
the graph (via graph_tools.py, bypassing the LLM entirely) so grading is
objective rather than LLM-graded. Checks whether the LLM's final answer
text contains the key facts from that ground truth (substring/number
checks), mirroring ../LMM_dive's evaluate.py precedent of a real,
if crude, automatic scoring function instead of vibes-based grading.

No standard benchmark exists for this exact task (compositional NL
queries over a self-built multi-camera event graph), so this is a
self-verified eval set, same approach as ../LMM_dive's
check_stage1_grounding.py / check_stage2_instructions.py.

Usage:
  python scripts/eval_query_interface.py --graph data/scene_061/event_graph_with_attrs.gpickle
"""
import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from graph_tools import GraphTools
from query_interface import (SYSTEM_PROMPT, TOOL_SCHEMAS, _extract_tool_calls, _force_final_answer,
                              DETERMINISTIC_REPORT_TOOLS, DETERMINISTIC_REPORT_FORMATTERS, _combine_answer,
                              _mentions_unexecuted_tool)


def build_test_cases(tools: GraphTools) -> list:
    cases = []

    # 1. Exact-count sanity check against a value computed directly from the graph.
    multi = tools.list_multi_camera_entities()
    cases.append({
        "query": "How many distinct entities were confirmed to appear in more than one camera?",
        "check": lambda ans: str(len(multi)) in ans,
        "expected": str(len(multi)),
    })

    # 2. Timeline lookup for a specific known multi-camera entity.
    gid = multi[0]["global_id"]
    cams = set(multi[0]["cameras"])
    cases.append({
        "query": f"In which cameras did entity {gid} appear? List the camera numbers.",
        "check": lambda ans, cams=cams: all(c in ans for c in cams),
        "expected": f"cameras {sorted(cams)}",
    })

    # 3. Single-camera entity: correctly should NOT be reported as multi-camera.
    single_cam_entities = [
        d["global_id"] for n, d in tools.G.nodes(data=True) if d["type"] == "entity"
    ]
    single_gid = next(gid2 for gid2 in single_cam_entities if gid2 not in {m["global_id"] for m in multi})
    cases.append({
        "query": f"Was entity {single_gid} seen in more than one camera? Answer yes or no and explain.",
        "check": lambda ans: bool(re.search(r"\bno\b", ans, re.IGNORECASE)),
        "expected": "no",
    })

    # 4. Nonexistent entity -- should trigger an explicit "don't know" / error response, not a fabrication.
    fake_gid = max(single_cam_entities) + 1000
    cases.append({
        "query": f"Tell me about entity {fake_gid}'s movements.",
        "check": lambda ans: any(w in ans.lower() for w in ["no entity", "not exist", "no record", "no data", "don't have", "no evidence", "not found", "no information"]),
        "expected": "should report no such entity found",
    })

    # 5. Appearance search sanity: does search surface an entity whose actual caption matches.
    caption_entities = [(gid2, d["appearance_caption"]) for gid2, d in tools.entities.items() if "skateboard" in d.get("appearance_caption", "")]
    if caption_entities:
        target_gid, target_caption = caption_entities[0]
        cases.append({
            "query": "Find a person with a skateboard and tell me their global_id.",
            "check": lambda ans, gids={g for g, _ in caption_entities}: any(str(g) in ans for g in gids),
            "expected": f"one of global_ids {[g for g,_ in caption_entities]}",
        })

    # 6. Camera+time window ground truth (count of entities present).
    cam_entities = tools.find_entities_in_camera("535", 0, 5)
    cases.append({
        "query": "Which entities were present in camera 535 during the first 5 seconds? Give their global_ids.",
        "check": lambda ans, gids={e["global_id"] for e in cam_entities}: sum(str(g) in ans for g in gids) >= max(1, len(gids) // 2),
        "expected": f"global_ids {[e['global_id'] for e in cam_entities]}",
    })

    # 7. Real spatial-proximity ground truth (distance-based, not just camera+time co-occurrence --
    # this is the axis check_entities_cooccur alone can't answer; see find_nearby_entities).
    all_gids = [d["global_id"] for n, d in tools.G.nodes(data=True) if d["type"] == "entity"]
    for candidate_gid in [m["global_id"] for m in multi] + all_gids:
        nearby = tools.find_nearby_entities(candidate_gid, max_distance_m=2.0, max_gap_sec=1.0)
        all_nearby = (nearby["confirmed_proximity_matches"] + nearby["uncertain_proximity_matches"]
                      + nearby["likely_same_person_matches"])
        if all_nearby:
            cases.append({
                "query": f"Which entities came within 2 meters of entity {candidate_gid} at any point? Give their global_ids.",
                "check": lambda ans, gids={n["global_id"] for n in all_nearby}: any(str(g) in ans for g in gids),
                "expected": f"one of global_ids {[n['global_id'] for n in all_nearby]}",
            })
            break

    return cases


def run_query(tokenizer, model, tools, tool_fns, query: str, max_iters: int = 15) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": query},
    ]
    deterministic_reports = []
    nudges_remaining = 2
    for _ in range(max_iters):
        prompt = tokenizer.apply_chat_template(messages, tools=TOOL_SCHEMAS, add_generation_prompt=True, tokenize=False)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out_ids = model.generate(**inputs, max_new_tokens=1024, do_sample=False)
        response_text = tokenizer.decode(out_ids[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
        calls = _extract_tool_calls(response_text)
        if not calls:
            unexecuted = _mentions_unexecuted_tool(response_text, tool_fns.keys())
            if unexecuted and nudges_remaining > 0:
                nudges_remaining -= 1
                messages.append({"role": "assistant", "content": response_text})
                messages.append({"role": "user", "content":
                    f"You described calling {unexecuted} but did not actually call it. To call a tool "
                    f"you MUST emit a real <tool_call>{{\"name\": ..., \"arguments\": ...}}</tool_call> "
                    f"block -- describing it in words does not call it. Call it now."})
                continue
            answer, _ = _combine_answer(response_text, deterministic_reports)
            return answer
        messages.append({"role": "assistant", "content": response_text})
        for call in calls:
            name, call_args = call["name"], call.get("arguments", {})
            try:
                result = tool_fns[name](**call_args) if name in tool_fns else {"error": f"unknown tool {name}"}
            except Exception as e:
                result = {"error": str(e)}
            if name in DETERMINISTIC_REPORT_TOOLS and not (isinstance(result, dict) and "error" in result):
                report = DETERMINISTIC_REPORT_FORMATTERS[name](result)
                header = f"--- {name}({', '.join(f'{k}={v!r}' for k, v in call_args.items())}) ---"
                deterministic_reports.append(f"{header}\n{report}")
                messages.append({"role": "tool", "name": name, "content": report})
            else:
                messages.append({"role": "tool", "name": name, "content": json.dumps(result)})
    forced = _force_final_answer(tokenizer, model, messages, verbose=False)
    answer, _ = _combine_answer(forced, deterministic_reports)
    return answer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", type=str, default=str(PROJECT_ROOT / "data" / "scene_061" / "event_graph_with_attrs.gpickle"))
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--out", type=str, default=str(PROJECT_ROOT / "data" / "scene_061" / "query_eval_results.json"))
    args = parser.parse_args()

    tools = GraphTools(args.graph)
    tool_fns = {
        "search_by_appearance": tools.search_by_appearance,
        "get_entity_timeline": tools.get_entity_timeline,
        "find_entities_in_camera": tools.find_entities_in_camera,
        "check_entities_cooccur": tools.check_entities_cooccur,
        "find_nearby_entities": tools.find_nearby_entities,
        "find_nearby_entities_by_description": tools.find_nearby_entities_by_description,
        "list_multi_camera_entities": tools.list_multi_camera_entities,
        "rank_entities_by_interaction_count": tools.rank_entities_by_interaction_count,
    }

    cases = build_test_cases(tools)
    print(f"Built {len(cases)} test cases")

    print(f"Loading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    results = []
    n_pass = 0
    for i, case in enumerate(cases):
        print(f"\n[{i+1}/{len(cases)}] {case['query']}")
        answer = run_query(tokenizer, model, tools, tool_fns, case["query"])
        passed = case["check"](answer)
        n_pass += passed
        print(f"  expected: {case['expected']}")
        print(f"  answer: {answer[:200]}")
        print(f"  PASS" if passed else "  FAIL")
        results.append({"query": case["query"], "expected": case["expected"], "answer": answer, "passed": passed})

    print(f"\n=== {n_pass}/{len(cases)} passed ({100*n_pass/len(cases):.1f}%) ===")
    with open(args.out, "w") as f:
        json.dump({"n_pass": n_pass, "n_total": len(cases), "results": results}, f, indent=2)
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
