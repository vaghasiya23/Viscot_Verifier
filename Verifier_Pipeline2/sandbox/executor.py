"""
sandbox/executor.py - Safe execution environment for visual verification code.
"""
import traceback
import sys
from typing import Dict, Any
from tools.image_patch import ImagePatch
from tools.spatial import box_iou, box_distance, center_distance
from tools.math_verify import verify_math


def execute_verifier(code_str: str, image_path: str) -> Dict[str, Any]:
    """
    Executes generated verification code against the target image.
    Returns a structured verdict dictionary.
    """
    img = ImagePatch(image_path)

    import math
    import numpy as np
    import re

    # Safe execution scope with available tools
    exec_scope = {
        "ImagePatch": ImagePatch,
        "box_iou": box_iou,
        "box_distance": box_distance,
        "center_distance": center_distance,
        "verify_math": verify_math,
        "math": math,
        "np": np,
        "numpy": np,
        "re": re,
        "img": img,
        "__builtins__": __builtins__,
    }

    try:
        # Compile code to catch syntax errors immediately
        compiled = compile(code_str, "<verification_code>", "exec")
        exec(compiled, exec_scope)

        # Check if code defines a verify_reasoning function
        if "verify_reasoning" in exec_scope and callable(exec_scope["verify_reasoning"]):
            result = exec_scope["verify_reasoning"](img)
            if isinstance(result, dict) and "verdict" in result:
                # Ensure all required fields exist
                steps = result.get("steps", [])
                passed = result.get("passed_count", sum(1 for s in steps if s.get("passed", False)))
                failed_step = result.get("failed_step")

                # Compute total steps accurately (must account for failed_step on early exit)
                total = result.get("total_steps")
                if total is None:
                    if failed_step is not None:
                        total = max(len(steps), failed_step)
                    else:
                        total = len(steps) if len(steps) > 0 else 1

                result["total_steps"] = total
                result["passed_count"] = passed

                if result["verdict"] == "INVALID":
                    # An invalid verdict must have a completion ratio strictly < 1.0
                    ratio = passed / max(1, total)
                    result["completion_ratio"] = min(0.99, ratio) if total > 0 else 0.0
                else:
                    result["completion_ratio"] = passed / max(1, total) if total > 0 else 1.0
                return result

        # If executed sequentially as top-level assertions without throwing:
        return {
            "verdict": "VALID",
            "failed_step": None,
            "error": None,
            "passed_count": 1,
            "total_steps": 1,
            "completion_ratio": 1.0,
            "details": "All top-level assertions passed successfully."
        }

    except AssertionError as ae:
        err_msg = str(ae) if str(ae) else "Assertion failed"
        # Try to infer step number from error message if formatted e.g. "Step 2 Failed: ..."
        failed_step = None
        if "step" in err_msg.lower():
            import re
            m = re.search(r"step\s*(\d+)", err_msg, re.IGNORECASE)
            if m:
                failed_step = int(m.group(1))

        return {
            "verdict": "INVALID",
            "failed_step": failed_step,
            "error": err_msg,
            "passed_count": max(0, (failed_step - 1) if failed_step else 0),
            "total_steps": failed_step if failed_step else 1,
            "completion_ratio": 0.0 if not failed_step else (failed_step - 1) / failed_step,
            "traceback": traceback.format_exc(),
        }

    except Exception as e:
        return {
            "verdict": "ERROR",
            "failed_step": None,
            "error": f"{type(e).__name__}: {str(e)}",
            "passed_count": 0,
            "total_steps": 1,
            "completion_ratio": 0.0,
            "traceback": traceback.format_exc(),
        }
