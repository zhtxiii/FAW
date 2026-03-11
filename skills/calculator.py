import ast
import operator
from typing import Any
from skill_manager import Skill


class CalculatorSkill(Skill):
    """安全的数学表达式计算技能。"""

    name = "calculator"
    description = "计算数学表达式，输入 {'expression': '2 + 3 * 4'} 返回计算结果。"

    _ALLOWED_NODES = (
        ast.Expression, ast.BinOp, ast.UnaryOp, ast.Constant,
    )
    _OPS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.Mod: operator.mod,
        ast.FloorDiv: operator.floordiv,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    async def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        expr = params.get("expression", "")
        try:
            tree = ast.parse(expr, mode="eval")
            result = self._eval_node(tree.body)
            return {"result": result}
        except Exception as e:
            return {"error": str(e)}

    def _eval_node(self, node: ast.AST) -> float | int:
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp):
            op_func = self._OPS.get(type(node.op))
            if op_func is None:
                raise ValueError(f"不支持的运算: {type(node.op).__name__}")
            return op_func(self._eval_node(node.left), self._eval_node(node.right))
        if isinstance(node, ast.UnaryOp):
            op_func = self._OPS.get(type(node.op))
            if op_func is None:
                raise ValueError(f"不支持的一元运算: {type(node.op).__name__}")
            return op_func(self._eval_node(node.operand))
        raise ValueError(f"不允许的语句: {type(node).__name__}")
