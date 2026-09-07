"""Strict scaffold conformance: code must implement every trusted statement.

This intentionally rejects equivalent rewrites. Comments/formatting may vary;
arbitrary Python transformations cannot be proved evidence-preserving here.
"""
import ast
from generation.skeleton_gen import canonical_code


def check(code, scaffold):
    if not scaffold or any(s["needs_review"] for s in scaffold):
        return {"passed": False, "coverage": 0.0, "reason": "Unsupported scaffold requires review"}
    try:
        expected = ast.dump(ast.parse(canonical_code(scaffold)), include_attributes=False)
        actual = ast.dump(ast.parse(code), include_attributes=False)
    except SyntaxError as exc:
        return {"passed": False, "coverage": 0.0, "reason": f"Invalid Python: {exc.msg}"}
    passed = actual == expected
    return {"passed": passed, "coverage": 1.0 if passed else 0.0,
            "reason": "All scaffold statements preserved" if passed else
            "Code changed/omitted scaffold evidence, dependencies, assertions or answer derivation"}
