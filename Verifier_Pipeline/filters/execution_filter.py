"""
execution_filter.py

Runs a generated trace's code block against the real tool registry.
This is intentionally NOT a semantic filter -- "ran without crashing" is
necessary but not sufficient (see coverage_filter.py and
arg_validity_filter.py for the semantic checks that catch code which runs
clean but verified nothing meaningful).
"""
import signal
import traceback

from tools import TOOL_REGISTRY


class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):
    raise TimeoutError("Execution exceeded time limit")


def run(code: str, img_path: str, timeout_seconds: int = 30) -> dict:
    """
    Executes `code` with img_path bound and all tools available.
    Returns {"ok": bool, "local_scope": dict, "error": str|None}.
    local_scope lets downstream filters inspect what variables/values the
    code actually produced (e.g. `final_answer`).
    """
    if not code:
        return {"ok": False, "local_scope": {}, "error": "No python code block found"}

    local_scope = dict(TOOL_REGISTRY)
    local_scope["img_path"] = img_path

    signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_seconds)
    try:
        exec(code, {"__builtins__": __builtins__}, local_scope)
        return {"ok": True, "local_scope": local_scope, "error": None}
    except AssertionError as e:
        # Assertion failures are a VALID outcome for INVALID-verdict traces
        # -- not necessarily a bad trace, just one that correctly caught a
        # broken claim. Caller (pipeline.py) decides what to do with this
        # based on whether the trace's own <final_verdict> says INVALID.
        return {"ok": False, "local_scope": local_scope, "error": f"AssertionError: {e}", "is_assertion": True}
    except Exception as e:
        return {"ok": False, "local_scope": local_scope, "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}"}
    finally:
        signal.alarm(0)