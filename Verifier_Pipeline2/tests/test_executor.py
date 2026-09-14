"""
tests/test_executor.py - Unit test for the execution sandbox.
"""
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sandbox import execute_verifier

def test_sandbox():
    img_path = os.path.join(ROOT, "data", "raw", "images", "2410353.jpg")
    assert os.path.exists(img_path)

    # Test Case 1: Valid Code
    valid_code = """
def verify_reasoning(img) -> dict:
    trace = []
    # Step 1: Detect chair
    chairs = img.find("chair")
    if len(chairs) == 0:
        return {"verdict": "INVALID", "failed_step": 1, "error": "No chair", "steps": trace}
    trace.append({"step": 1, "claim": "chair exists", "passed": True})

    # Step 2: Detect sofa
    sofas = img.find("sofa")
    if len(sofas) == 0:
        return {"verdict": "INVALID", "failed_step": 2, "error": "No sofa", "steps": trace}
    trace.append({"step": 2, "claim": "sofa exists", "passed": True})

    # Step 3: Check spatial relation (sofa to the right of chair)
    chair = chairs[0]
    right_sofas = [s for s in sofas if s.horizontal_center > chair.horizontal_center]
    if len(right_sofas) == 0:
        return {"verdict": "INVALID", "failed_step": 3, "error": "No sofa to the right", "steps": trace}
    trace.append({"step": 3, "claim": "sofa to right of chair", "passed": True})

    return {"verdict": "VALID", "failed_step": None, "steps": trace}
"""

    print("\n--- Running Valid Code Test ---")
    result_valid = execute_verifier(valid_code, img_path)
    print(f"Valid Code Result: {result_valid}")
    assert result_valid["verdict"] == "VALID"
    assert result_valid["completion_ratio"] == 1.0
    assert result_valid["passed_count"] == 3

    # Test Case 2: Invalid Code (Hallucination)
    invalid_code = """
def verify_reasoning(img) -> dict:
    trace = []
    # Step 1: Detect non-existent elephant
    elephants = img.find("elephant")
    if len(elephants) == 0:
        return {"verdict": "INVALID", "failed_step": 1, "error": "No elephant detected", "steps": trace}
    trace.append({"step": 1, "claim": "elephant exists", "passed": True})
    return {"verdict": "VALID", "failed_step": None, "steps": trace}
"""

    print("\n--- Running Invalid Code Test (Hallucination Catch) ---")
    result_invalid = execute_verifier(invalid_code, img_path)
    print(f"Invalid Code Result: {result_invalid}")
    assert result_invalid["verdict"] == "INVALID"
    assert result_invalid["failed_step"] == 1
    assert result_invalid["completion_ratio"] == 0.0

    # Test Case 3: Partial Failure (Step 1 passes, Step 2 fails)
    partial_fail_code = """
def verify_reasoning(img) -> dict:
    trace = []
    # Step 1: Detect chair (passes)
    chairs = img.find("chair")
    trace.append({"step": 1, "claim": "chair exists", "passed": True})

    # Step 2: Use math and fail on step 2
    if not verify_math("2 + 2 == 5"):
        return {"verdict": "INVALID", "failed_step": 2, "error": "Math failed", "steps": trace}
    trace.append({"step": 2, "claim": "math holds", "passed": True})
    return {"verdict": "VALID", "failed_step": None, "steps": trace}
"""
    print("\n--- Running Partial Failure Test ---")
    result_partial = execute_verifier(partial_fail_code, img_path)
    print(f"Partial Failure Result: {result_partial}")
    assert result_partial["verdict"] == "INVALID"
    assert result_partial["failed_step"] == 2
    assert result_partial["passed_count"] == 1
    assert result_partial["total_steps"] == 2
    assert abs(result_partial["completion_ratio"] - 0.5) < 1e-4

    print("\n--- All Sandbox Tests Passed! ---")

if __name__ == "__main__":
    test_sandbox()
