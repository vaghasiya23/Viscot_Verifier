"""Conservative answer agreement; a model verdict alone proves nothing."""
import re

ALIASES = ({"sofa", "couch"}, {"cabinet", "cupboard"}, {"pants", "trousers"},
           {"child", "children"}, {"hot dog", "hotdog"})


def normalize(answer):
    return " ".join(str(answer).strip().lower().rstrip(".!?").split())


def are_answers_compatible(derived, ground_truth):
    d, gt = normalize(derived), normalize(ground_truth)
    return bool(d and gt) and (d == gt or any(d in group and gt in group for group in ALIASES))


def check(exec_result, sample, verdict_text):
    if not exec_result.get("ok"):
        return {"status": "failed", "reason": "Execution did not verify all claims"}
    if not re.match(r"^VALID\b", verdict_text.strip(), re.I) or re.search(r"\bINVALID\b", verdict_text, re.I):
        return {"status": "failed", "reason": "No unambiguous VALID verdict; requires review"}
    answer = exec_result.get("local_scope", {}).get("final_answer")
    events = exec_result.get("evidence", [])
    if not any(e["tool"] == "vlm_query" and e.get("result") == answer for e in events):
        return {"status": "failed", "reason": "Answer lacks a recorded visual query"}
    if answer is None or not are_answers_compatible(answer, sample.get("answer", "")):
        return {"status": "failed", "reason": f"Visual answer {answer!r} disagrees with target"}
    return {"status": "valid", "reason": "Complete execution and visual answer agreement"}
