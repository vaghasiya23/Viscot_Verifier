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


def run(code: str, img_path: str, timeout_seconds: int = 120) -> dict:
    """
    Executes `code` with img_path bound and all tools available.
    Returns {"ok": bool, "local_scope": dict, "error": str|None}.
    local_scope lets downstream filters inspect what variables/values the
    code actually produced (e.g. `final_answer`).
    """
    if not code:
        return {"ok": False, "local_scope": {}, "error": "No python code block found"}

    evidence = []
    def record(name, tool):
        def wrapped(*args, **kwargs):
            if len(evidence) >= 64:
                raise RuntimeError("Uncertain: tool-call budget exceeded")
            event = {"tool": name, "args": args, "kwargs": kwargs}
            evidence.append(event)
            try:
                result = tool(*args, **kwargs)
                event["result"] = result
                return result
            except Exception as exc:
                event["error"] = f"{type(exc).__name__}: {exc}"
                raise
        return wrapped
    local_scope = {name: record(name, tool) for name, tool in TOOL_REGISTRY.items()}
    local_scope["img_path"] = img_path

    previous_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_seconds)
    try:
        exec(code, local_scope, local_scope)
        return {"ok": True, "local_scope": local_scope, "evidence": evidence, "error": None}
    except AssertionError as e:
        # Failed verification is not proof that the source claim is false.
        return {"ok": False, "local_scope": local_scope, "evidence": evidence, "error": f"AssertionError: {e}", "is_assertion": True}
    except Exception as e:
        return {"ok": False, "local_scope": local_scope, "evidence": evidence, "error": f"{type(e).__name__}: {e}\n{traceback.format_exc()}"}
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)