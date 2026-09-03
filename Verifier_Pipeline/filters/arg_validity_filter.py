"""
arg_validity_filter.py

Checks that tool calls in the generated code use arguments that match
the documented tool signatures. Catches semantic garbage like:
  detect_and_crop(img_path, "overlapping")   # "overlapping" is a relation, not an object
  check_spatial_relation(box1, box2, "wearing")  # "wearing" is semantic, not spatial

Uses regex to extract tool calls from the code string and validates
their arguments against known-good patterns.
"""
import re

# Valid spatial relations for check_spatial_relation
VALID_SPATIAL_RELATIONS = {
    "overlapping", "inside", "left", "right", "above", "below", "near"
}

# Words that are clearly relations/predicates, NOT object classes.
# If these show up as the object_query in detect_and_crop, flag it.
RELATION_WORDS = {
    "overlapping", "inside", "left", "right", "above", "below", "near",
    "in_front_of", "behind", "wearing", "holding", "riding", "eating",
    "sitting", "standing", "lying", "sleeping", "on", "in", "under",
}


def check(code: str) -> dict:
    """
    Args:
        code: the extracted python code block from the trace

    Returns:
        {"passed": bool, "reason": str, "violations": list}
    """
    if not code:
        return {"passed": False, "reason": "No code to check", "violations": []}

    violations = []

    # Check 1: detect_and_crop called with a relation word instead of an object
    detect_calls = re.findall(
        r'detect_and_crop\s*\(\s*\w+\s*,\s*["\']([^"\']+)["\']\s*\)', code
    )
    for query in detect_calls:
        query_lower = query.strip().lower()
        if query_lower in RELATION_WORDS:
            violations.append(
                f"detect_and_crop called with relation word '{query}' as object_query — "
                f"this is semantically void (detector will find nothing meaningful)"
            )

    # Check 2: check_spatial_relation called with an invalid relation string
    spatial_calls = re.findall(
        r'check_spatial_relation\s*\([^,]+,[^,]+,\s*["\']([^"\']+)["\']\s*\)', code
    )
    for rel in spatial_calls:
        rel_lower = rel.strip().lower()
        if rel_lower not in VALID_SPATIAL_RELATIONS:
            violations.append(
                f"check_spatial_relation called with unsupported relation '{rel}' — "
                f"valid options: {VALID_SPATIAL_RELATIONS}. "
                f"Use vlm_probe for semantic relations like '{rel}'."
            )

    # Check 3: Forbidden imports (model should use only the provided tools)
    import_matches = re.findall(r'^\s*(?:import|from)\s+(\w+)', code, re.MULTILINE)
    forbidden = {"cv2", "numpy", "torch", "PIL", "pytesseract", "transformers"}
    for mod in import_matches:
        if mod in forbidden:
            violations.append(
                f"Code imports '{mod}' — generated code must use ONLY the provided "
                f"tool API, not raw library calls."
            )

    passed = len(violations) == 0
    reason = "All tool arguments valid" if passed else f"{len(violations)} violation(s) found"

    return {"passed": passed, "reason": reason, "violations": violations}