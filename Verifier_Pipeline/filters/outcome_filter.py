"""
outcome_filter.py

Checks whether the trace's derived answer is consistent with the sample's
ground-truth answer. This is Filter #2 from the plan:
  "the trajectory's derived answer actually matches sample.answer"

Two checks:
1. If the code set a `final_answer` variable, compare it against sample['answer'].
2. If the <final_verdict> says VALID, verify that the ground-truth answer
   string appears somewhere in the verdict text (sanity check that the model
   actually compared against it, not just rubber-stamped).
"""


def check(exec_result: dict, sample: dict, verdict_text: str) -> dict:
    """
    Args:
        exec_result: from execution_filter.run() — contains 'ok', 'local_scope', 'error'
        sample: the original data sample with 'answer' field
        verdict_text: extracted <final_verdict> text from the trace

    Returns:
        {"passed": bool, "reason": str}
    """
    gt_answer = sample.get("answer", "").strip().lower()
    if not gt_answer:
        return {"passed": False, "reason": "No ground-truth answer in sample"}

    # Check 1: If code produced a final_answer variable, does it match?
    local_scope = exec_result.get("local_scope", {})
    code_answer = local_scope.get("final_answer", None)
    if code_answer is not None:
        code_answer_str = str(code_answer).strip().lower()
        if gt_answer not in code_answer_str and code_answer_str not in gt_answer:
            return {
                "passed": False,
                "reason": f"Code's final_answer='{code_answer_str}' doesn't match ground truth='{gt_answer}'"
            }

    # Check 2: If verdict says VALID, the ground-truth answer should appear
    # somewhere in the verdict (proves the model actually compared)
    verdict_lower = verdict_text.lower()
    if "valid" in verdict_lower and "invalid" not in verdict_lower:
        # It claims VALID — loose check that it referenced the answer
        if gt_answer not in verdict_lower and len(gt_answer) > 2:
            return {
                "passed": False,
                "reason": f"Verdict says VALID but never mentions ground-truth answer '{gt_answer}'"
            }

    # Check 3: If verdict says INVALID, that's a signal this CoT is wrong —
    # which is a valid outcome! We pass it but flag it differently.
    if "invalid" in verdict_lower:
        return {
            "passed": True,
            "reason": "Verdict is INVALID — trace correctly identified a broken claim",
            "is_invalidation": True
        }

    return {"passed": True, "reason": "Outcome consistent with ground truth"}