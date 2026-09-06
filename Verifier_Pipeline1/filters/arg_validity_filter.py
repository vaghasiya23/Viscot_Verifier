"""
arg_validity_filter.py

Validates generated Python code before execution:
  1. No pure relation words or pronouns passed as detect_and_crop queries.
  2. detect_and_crop must take a single string literal, not a list.
  3. check_spatial_relation must use valid 2D spatial relation strings.
  4. No raw library imports (cv2, torch, etc.).
"""
import re

VALID_SPATIAL_RELATIONS = {
    "overlapping", "inside", "left", "right", "above", "below", "near",
    "in", "on", "next to", "beside", "under", "beneath"
}

RELATION_WORDS = {
    "overlapping", "inside", "left", "right", "above", "below", "near",
    "in_front_of", "behind", "wearing", "holding", "riding", "eating",
    "sitting", "standing", "lying", "sleeping", "under",
}

NON_NOUN_WORDS = {
    "he", "she", "it", "they", "this", "that", "one", "who", "what",
    "which", "him", "her", "them", "someone", "something",
}

FORBIDDEN_IMPORTS = {"cv2", "numpy", "torch", "PIL", "pytesseract", "transformers"}


def check(code: str) -> dict:
    if not code:
        return {"passed": False, "reason": "No code to check", "violations": []}

    violations = []

    # --- Check 1: relation word or pronoun used as detect_and_crop query ---
    detect_calls_str = re.findall(
        r'detect_and_crop\s*\(\s*\w+\s*,\s*["\']([^"\']+)["\']\s*\)', code
    )
    for query in detect_calls_str:
        q = query.strip().lower()
        if q in RELATION_WORDS:
            violations.append(
                f"detect_and_crop called with relation word '{query}' as object_query -- "
                f"semantically void, detector will find nothing meaningful"
            )
        if q in NON_NOUN_WORDS:
            violations.append(
                f"detect_and_crop called with pronoun/non-noun '{query}' as object_query -- "
                f"not a detectable visual class; resolve the referent to an actual noun"
            )

    # --- Check 2: detect_and_crop called with a list instead of a string ---
    list_arg_calls = re.findall(r'detect_and_crop\s*\(\s*\w+\s*,\s*(\[[^\]]*\])\s*\)', code)
    for bad_arg in list_arg_calls:
        violations.append(
            f"detect_and_crop called with a list argument {bad_arg} -- the tool signature "
            f"takes a single string object_query, not a list."
        )

    # --- Check 3: check_spatial_relation with invalid relation string ---
    spatial_calls = re.findall(
        r'check_spatial_relation\s*\([^,]+,[^,]+,\s*["\']([^"\']+)["\']\s*\)', code
    )
    for rel in spatial_calls:
        rel_lower = rel.strip().lower().replace("_", " ")
        is_valid = any(v in rel_lower for v in VALID_SPATIAL_RELATIONS)
        if not is_valid:
            violations.append(
                f"check_spatial_relation called with unsupported relation '{rel}' -- "
                f"valid options: {VALID_SPATIAL_RELATIONS}."
            )

    # --- Check 4: forbidden raw library imports ---
    import_matches = re.findall(r'^\s*(?:import|from)\s+(\w+)', code, re.MULTILINE)
    for mod in import_matches:
        if mod in FORBIDDEN_IMPORTS:
            violations.append(
                f"Code imports '{mod}' -- generated code must use ONLY the provided "
                f"tool API, not raw library calls."
            )

    passed = len(violations) == 0
    reason = "All tool arguments valid" if passed else f"{len(violations)} violation(s) found"
    return {"passed": passed, "reason": reason, "violations": violations}
