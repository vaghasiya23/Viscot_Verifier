"""
coverage_filter.py (FIXED)

Bug fixed: query steps were counted "covered" if the code contained
EITHER "final_answer" OR "assert" anywhere -- and nearly every trace has
an assert somewhere for unrelated reasons, so this check almost never
actually failed a trace on missing query coverage. Now requires
final_answer specifically for query steps.
"""


def check(code: str, scaffold: list) -> dict:
    if not code:
        return {"passed": False, "reason": "No code to check", "coverage": 0.0}

    total_steps = len(scaffold)
    if total_steps == 0:
        return {"passed": True, "reason": "No scaffold steps to verify", "coverage": 1.0}

    covered = 0
    missing = []

    for step in scaffold:
        var_name = step["var_name"]
        op = step["operation"]

        if op == "query":
            # query steps must produce final_answer specifically -- "assert"
            # existing elsewhere in the code proves nothing about this step.
            if "final_answer" in code:
                covered += 1
            else:
                missing.append(f"step {step['step_index']} ({op}) -- no final_answer set")
        elif var_name in code:
            covered += 1
        else:
            label = step.get("parsed", {}).get("label") or step.get("parsed", {}).get("predicate", "?")
            missing.append(f"step {step['step_index']} ({op}: {label})")

    coverage = covered / total_steps if total_steps > 0 else 0.0
    threshold = 0.8
    passed = coverage >= threshold

    reason = f"Coverage: {covered}/{total_steps} ({coverage:.0%})"
    if missing:
        reason += f". Missing: {', '.join(missing)}"

    return {"passed": passed, "reason": reason, "coverage": coverage}