"""
outcome_filter.py

Checks whether the trace's derived answer is consistent with the ground-truth answer.
Supports common-sense semantic matching (synonyms, singular/plural, category entailment).
"""

# Common-sense semantic equivalences and category entailment
SEMANTIC_SYNONYMS = {
    "children": {"child", "children", "kid", "kids", "person", "people", "boy", "boys", "girl", "girls"},
    "child": {"child", "children", "kid", "kids", "person", "people"},
    "person": {"person", "people", "man", "men", "woman", "women", "child", "children", "boy", "girl"},
    "sofa": {"sofa", "couch", "settee", "furniture"},
    "couch": {"sofa", "couch", "furniture"},
    "cabinet": {"cabinet", "cupboard", "shelf", "furniture"},
    "rock": {"rock", "stone", "boulder"},
    "hot dog": {"hot dog", "hotdog", "frankfurter", "food", "fast food"},
    "pants": {"pants", "trousers", "jeans", "slacks", "clothing", "apparel"},
    "shirt": {"shirt", "t-shirt", "top", "clothing", "apparel"},
    "bird": {"bird", "animal"},
    "horse": {"horse", "animal"},
}


def are_answers_compatible(derived: str, ground_truth: str) -> bool:
    """Checks exact match, substring inclusion, or common-sense semantic equivalence."""
    d = derived.strip().lower()
    gt = ground_truth.strip().lower()

    # Exact or substring match
    if d == gt or d in gt or gt in d:
        return True

    # Check synonym / entailment mapping
    if gt in SEMANTIC_SYNONYMS and d in SEMANTIC_SYNONYMS[gt]:
        return True
    if d in SEMANTIC_SYNONYMS and gt in SEMANTIC_SYNONYMS[d]:
        return True

    return False


def check(exec_result: dict, sample: dict, verdict_text: str) -> dict:
    """
    Returns {"status": "valid" | "correctly_invalidated" | "failed", "reason": str}
    """
    gt_answer = str(sample.get("answer", "")).strip().lower()
    if not gt_answer:
        return {"status": "failed", "reason": "No ground-truth answer in sample"}

    verdict_lower = verdict_text.lower()
    says_valid = "invalid" not in verdict_lower and "valid" in verdict_lower
    says_invalid = "invalid" in verdict_lower

    local_scope = exec_result.get("local_scope", {})
    code_answer = local_scope.get("final_answer", None)

    if says_invalid:
        return {
            "status": "correctly_invalidated",
            "reason": "Verdict is INVALID -- trace correctly identified a broken claim. Route to invalidated bucket, not golden.",
        }

    if not says_valid:
        return {"status": "failed", "reason": "Verdict text doesn't clearly state VALID or INVALID"}

    if code_answer is None:
        return {"status": "failed", "reason": "Verdict says VALID but code never set final_answer"}

    code_answer_str = str(code_answer).strip().lower()
    if not are_answers_compatible(code_answer_str, gt_answer):
        return {
            "status": "failed",
            "reason": f"Verdict says VALID but final_answer='{code_answer_str}' is not compatible with ground truth='{gt_answer}'",
        }

    return {"status": "valid", "reason": "VALID verdict, final_answer matches ground truth"}
