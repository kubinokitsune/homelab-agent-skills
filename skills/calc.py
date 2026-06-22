"""A safe scientific/engineering calculator -- deterministic, not LLM arithmetic.

Local models are unreliable at arithmetic, so the actual numbers are computed
here by evaluating a math expression through Python's ``ast`` with a strict
whitelist (numbers, operators, math functions, physical constants only -- no
attribute access, no arbitrary calls). An agent may use the LLM to *translate*
a worded question into an expression, but the value always comes from this.

    calc.evaluate("sqrt(2*g*1.5)")   -> Result(5.42...)
    calc.evaluate("R*298")           -> Result(2477.7...)

Constants (SI): pi, e (Euler), tau, g, c, G, h, hbar, kB (Boltzmann), Na
(Avogadro), R (gas), qe (elementary charge), eps0, mu0, atm, Far (Faraday).
"""

from __future__ import annotations

import ast
import math
import operator

from skills.errors import skill
from skills.logging import get_logger
from skills.result import Result

log = get_logger(__name__)

_FUNCS = {
    "sqrt": math.sqrt, "cbrt": lambda x: math.copysign(abs(x) ** (1 / 3), x),
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan, "atan2": math.atan2,
    "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
    "log": math.log10, "ln": math.log, "log10": math.log10, "log2": math.log2,
    "exp": math.exp, "abs": abs, "round": round, "floor": math.floor,
    "ceil": math.ceil, "factorial": math.factorial, "gcd": math.gcd,
    "radians": math.radians, "degrees": math.degrees, "rad": math.radians,
    "deg": math.degrees, "pow": pow, "min": min, "max": max, "hypot": math.hypot,
}

_NAMES = {
    "pi": math.pi, "e": math.e, "tau": math.tau, "inf": math.inf,
    # physical / engineering constants (SI)
    "g": 9.80665, "c": 299792458.0, "G": 6.67430e-11, "h": 6.62607015e-34,
    "hbar": 1.054571817e-34, "kB": 1.380649e-23, "Na": 6.02214076e23,
    "R": 8.314462618, "qe": 1.602176634e-19, "eps0": 8.8541878128e-12,
    "mu0": 1.25663706212e-6, "atm": 101325.0, "Far": 96485.33212,
}

_BIN = {
    ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
    ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
}
_UNARY = {ast.UAdd: operator.pos, ast.USub: operator.neg}


def _ev(node):
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise ValueError("only numbers are allowed")
        return node.value
    if isinstance(node, ast.BinOp):
        op = _BIN.get(type(node.op))
        if op is None:
            raise ValueError("operator not allowed")
        return op(_ev(node.left), _ev(node.right))
    if isinstance(node, ast.UnaryOp):
        op = _UNARY.get(type(node.op))
        if op is None:
            raise ValueError("unary operator not allowed")
        return op(_ev(node.operand))
    if isinstance(node, ast.Name):
        if node.id in _NAMES:
            return _NAMES[node.id]
        raise ValueError(f"unknown name '{node.id}'")
    if isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name) or node.func.id not in _FUNCS:
            raise ValueError("function not allowed")
        return _FUNCS[node.func.id](*[_ev(a) for a in node.args])
    raise ValueError("expression not allowed")


def fmt(value) -> str:
    """Format a result to ~6 significant figures (scientific when extreme)."""
    if isinstance(value, int):
        return str(value)
    return f"{value:.6g}"


@skill
def evaluate(expr: str) -> Result:
    """Evaluate a math expression deterministically. ``^`` is treated as power."""
    cleaned = expr.strip().replace("^", "**")
    tree = ast.parse(cleaned, mode="eval")
    value = _ev(tree.body)
    return Result.success(value)
