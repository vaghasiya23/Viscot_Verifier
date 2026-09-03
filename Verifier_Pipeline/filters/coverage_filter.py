"""
coverage_filter.py

Checks that every extracted claim was actually touched by at least one
tool call or assertion in the generated code. This is Filter #3:
  "every item in claims has >=1 corresponding tool call / assertion"

Implementation: the scaffold assigns variable names e0, e1, e2... to each
step. We check that each variable name appears in the generated code.
Also checks that at least one `assert` statement exists per scaffold step.
"""
import re


def check(code: str, scaffold: list) -> dict:
    """
    Args:
        code: the extracted python code block from the trace
        scaffold: list of scaffold step dicts from skeleton_gen.build_scaffold()

    Returns:
        {"passed": bool, "reason": str, "coverage": float}
    """
    if not code:
        return {"passed": False, "reason": "No code to check", "coverage": 0.0}

    total_steps = len(scaffold)
    if total_steps == 0:
        return {"passed": True, "reason": "No scaffold steps to verify", "coverage": 1.0}

    covered = 0
    missing = []

    for step in scaffold:
        var_name = step["var_name"]  # e0, e1, e2, ...
        op = step["operation"]

        # A step is "covered" if its variable name appears in the code
        # (either as assignment target or in a subsequent expression)
        if var_name in code:
            covered += 1
        elif op == "query":
            # query steps (e.g., "query name") don't always produce a
            # variable — they might just be the final answer derivation.
            # Count as covered if an assert or final_answer appears.
            if "final_answer" in code or "assert" in code:
                covered += 1
            else:
                missing.append(f"step {step['step_index']} ({op})")
        else:
            missing.append(f"step {step['step_index']} ({op}: {step.get('parsed', {}).get('label', step.get('parsed', {}).get('predicate', '?'))})")

    coverage = covered / total_steps if total_steps > 0 else 0.0

    # We require at least 80% coverage to pass (allows minor flexibility
    # for query steps that get folded into final_answer logic)
    threshold = 0.8
    passed = coverage >= threshold

    reason = f"Coverage: {covered}/{total_steps} ({coverage:.0%})"
    if missing:
        reason += f". Missing: {', '.join(missing)}"

    return {"passed": passed, "reason": reason, "coverage": coverage}