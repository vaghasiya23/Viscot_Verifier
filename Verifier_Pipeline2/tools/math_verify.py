"""
tools/math_verify.py - Mathematical expression verification using SymPy.
Used to verify arithmetic, algebraic, and geometric claims in CoT reasoning.
"""
import math as _math


def verify_math(expression: str) -> bool:
    """
    Verifies a mathematical expression or equation.

    Examples:
        verify_math("3 * 4 == 12")       → True
        verify_math("2 + 3 == 6")        → False
        verify_math("sqrt(16) == 4")     → True
        verify_math("abs(-5) == 5")      → True
        verify_math("round(3.7) == 4")   → True

    First tries SymPy for symbolic verification.
    Falls back to safe eval with math functions if SymPy fails.
    """
    # Attempt 1: SymPy symbolic evaluation
    try:
        import sympy

        result = sympy.sympify(expression)
        if isinstance(
            result,
            (bool, sympy.logic.boolalg.BooleanTrue, sympy.logic.boolalg.BooleanFalse),
        ):
            return bool(result)
        # For numeric results, check truthiness
        return bool(result)
    except Exception:
        pass

    # Attempt 2: Safe eval with restricted builtins
    _safe_globals = {"__builtins__": {}}
    _safe_locals = {
        "sqrt": _math.sqrt,
        "abs": abs,
        "pow": pow,
        "sin": _math.sin,
        "cos": _math.cos,
        "tan": _math.tan,
        "pi": _math.pi,
        "e": _math.e,
        "log": _math.log,
        "log2": _math.log2,
        "log10": _math.log10,
        "ceil": _math.ceil,
        "floor": _math.floor,
        "round": round,
        "min": min,
        "max": max,
        "sum": sum,
        "int": int,
        "float": float,
        "True": True,
        "False": False,
    }
    try:
        return bool(eval(expression, _safe_globals, _safe_locals))
    except Exception:
        return False
