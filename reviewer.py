"""FAW 审核守门员模块。

独立于执行逻辑。提供多种审核策略：
- SchemaReviewer：通过 JSON Schema 验证数据格式
- UnifiedValidator：代码优先 + LLM 兜底的统一验收器
- CompositeReviewer：组合多个审核器
"""

from __future__ import annotations

import json
import logging
import re
from abc import ABC, abstractmethod
from typing import Any

from llm_client import LLMClient
from models import ReviewResult, TaskRequest, TaskResult

logger = logging.getLogger(__name__)


class Reviewer(ABC):
    """审核器抽象基类。"""

    @abstractmethod
    async def review(self, task: TaskRequest, result: TaskResult) -> ReviewResult:
        """审核任务结果。

        Args:
            task: 原始任务请求。
            result: 智能体生成的结果。

        Returns:
            审核结果（通过/不通过 + 反馈）。
        """
        ...


class SchemaReviewer(Reviewer):
    """通过 JSON Schema 验证结果数据格式。

    简单实现：检查 expected_output_schema 中声明的必需字段是否存在。
    """

    async def review(self, task: TaskRequest, result: TaskResult) -> ReviewResult:
        schema = task.expected_output_schema
        if not schema:
            # 没有定义 schema 约束，直接通过
            return ReviewResult(passed=True, feedback="无 schema 约束，自动通过。")

        errors: list[str] = []

        # 检查 required 字段
        required_fields = schema.get("required", [])
        for field in required_fields:
            if field not in result.data:
                errors.append(f"缺少必需字段: {field}")

        # 检查 properties 定义中的类型
        properties = schema.get("properties", {})
        for key, spec in properties.items():
            if key in result.data:
                expected_type = spec.get("type")
                if expected_type and not self._type_matches(result.data[key], expected_type):
                    errors.append(
                        f"字段 '{key}' 类型不匹配: 期望 {expected_type}, "
                        f"实际 {type(result.data[key]).__name__}"
                    )

        if errors:
            return ReviewResult(passed=False, feedback="; ".join(errors))
        return ReviewResult(passed=True, feedback="Schema 验证通过。")

    @staticmethod
    def _type_matches(value: Any, expected: str) -> bool:
        type_map = {
            "string": str,
            "integer": int,
            "number": (int, float),
            "boolean": bool,
            "array": list,
            "object": dict,
        }
        py_type = type_map.get(expected)
        if py_type is None:
            return True  # 未知类型不做检查
        return isinstance(value, py_type)


class UnifiedValidator(Reviewer):
    """统一验收器：代码优先 + LLM 兜底。

    工作流程：
      1. 收集验证需求（自然语言，来自 validation_requirements 或向后兼容的 hardcode_rules）
      2. 调用专职代码生成 LLM 将自然语言翻译为 Python 验证代码
      3. 优先执行代码验证（可靠、确定性强）
      4. 对无法代码化的规则，兜底使用 LLM 语义审核
    """

    def __init__(self, llm: LLMClient | None = None):
        self._llm = llm or LLMClient()

    async def review(self, task: TaskRequest, result: TaskResult) -> ReviewResult:
        # 收集验证需求：优先使用新字段，向后兼容旧字段
        requirements = list(getattr(task, "validation_requirements", []))

        # 向后兼容：如果旧的 hardcode_rules 存在且是合法 Python 代码，直接作为代码规则执行
        legacy_rules = list(getattr(task, "hardcode_rules", []))

        if not requirements and not legacy_rules:
            return ReviewResult(passed=True, feedback="无验证需求，自动通过。")

        # ── Step 1: 处理旧式 hardcode_rules（向后兼容，直接当代码执行） ──
        legacy_errors = []
        if legacy_rules:
            legacy_errors = self._run_code_validation(legacy_rules, result.data, task.context)

        if legacy_errors:
            return ReviewResult(
                passed=False,
                feedback=f"[代码验证] {'; '.join(legacy_errors)}",
            )

        if not requirements:
            return ReviewResult(passed=True, feedback="旧式规则验证通过。")

        # ── Step 2: 调用代码生成模型翻译自然语言需求 ──
        synthesis = await self._synthesize_code_rules(
            requirements, task.expected_output_schema
        )

        code_rules = synthesis.get("code_rules", [])
        llm_fallback_rules = synthesis.get("llm_fallback_rules", [])

        # ── Step 3: 执行代码验证 ──
        code_errors = []
        if code_rules:
            code_errors = self._run_code_validation(code_rules, result.data, task.context)

        if code_errors:
            return ReviewResult(
                passed=False,
                feedback=f"[代码验证] {'; '.join(code_errors)}",
            )

        # ── Step 4: 无法代码化的规则走 LLM 兜底 ──
        if llm_fallback_rules:
            llm_result = await self._run_llm_validation(
                llm_fallback_rules, task, result
            )
            if not llm_result.passed:
                return ReviewResult(
                    passed=False,
                    feedback=f"[LLM 兜底验证] {llm_result.feedback}",
                )

        code_msg = f"代码验证 {len(code_rules)} 条通过" if code_rules else "无代码规则"
        llm_msg = f"LLM 兜底 {len(llm_fallback_rules)} 条通过" if llm_fallback_rules else "无需 LLM 兜底"
        return ReviewResult(
            passed=True,
            feedback=f"统一验收通过。{code_msg}；{llm_msg}。",
        )

    async def _synthesize_code_rules(
        self, requirements: list[str], schema: dict
    ) -> dict:
        """调用专职代码生成 LLM 将自然语言验证需求翻译为 Python 验证代码。

        返回格式：
          {"code_rules": ["Python expr 1", ...], "llm_fallback_rules": ["无法代码化的规则", ...]}
        """
        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个专职的验证代码生成工程师。你的任务是将自然语言验证需求翻译为 Python 验证代码片段。\n"
                    "严格要求：\n"
                    "1. 每条可代码化的规则必须是一段合法的 Python 代码（可以是单行表达式，也可以是多行逻辑分支）。你可以使用 import re, math 等安全库。\n"
                    "2. 如果你的代码是多行逻辑，你必须在代码最后明确赋值给一个变量 `is_valid`，其值为 True 或 False。如果是单行条件表达式则不需要。\n"
                    "3. 表达式中可使用的变量：`data`（dict，任务结果数据）、`context`（dict，任务上下文）。\n"
                    "4. 【致命规则】：你生成的代码中，所检查的字典键名（Keys）必须【绝对】【完全】参考我提供的 Schema！\n"
                    "   绝对不允许你自己主观臆造或乱翻译键名！（例如 Schema 里叫 source_url，你绝不能自作聪明写成 link）。\n"
                    "5. 【极度重要防死循环警告】：严禁写出判定条件过于死板和苛刻的规则！\n"
                    "   例如：不要硬核通过枚举公司名 `any(k in aff for k in ['nvidia', 'amd', 'intel', '华为'])` 来判断是不是硬件专家，只要包含相关字段或合理即可。\n"
                    "   如果某条需求过于模糊或主观（如「内容要有深度」「分析要全面」），无法用确定性代码表达，\n"
                    "   则将该需求原样放入 llm_fallback_rules 列表，**不要强行写弱智或致命严格的验证**。\n"
                    "6. 仅输出纯 JSON，格式为：{\"code_rules\": [\"Python code string\", ...], \"llm_fallback_rules\": [...]}\n"
                    "7. 不得输出任何额外文字、注释或 markdown 标记，仅输出合法 JSON 对象。\n"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"数据结构 Schema: {json.dumps(schema, ensure_ascii=False)}\n\n"
                    f"待翻译的自然语言验证需求列表：\n"
                    + "\n".join(f"- {r}" for r in requirements)
                ),
            },
        ]

        try:
            resp = await self._llm.chat_json(messages, temperature=0.1)
            # 确保返回值格式正确
            if isinstance(resp, dict):
                return {
                    "code_rules": resp.get("code_rules", []),
                    "llm_fallback_rules": resp.get("llm_fallback_rules", []),
                }
        except Exception as e:
            logger.warning("[验证代码合成] 合成失败，全部回退至 LLM 验证: %s", e)

        # 合成失败时，所有规则走 LLM 兜底
        return {"code_rules": [], "llm_fallback_rules": requirements}

    @staticmethod
    def _run_code_validation(
        code_rules: list[str], data: Any, context: dict
    ) -> list[str]:
        """执行 Python 验证代码。返回错误列表（空则全部通过）。"""
        errors = []
        import builtins
        
        eval_env = {
            "data": data,
            "context": context,
            "any": any,
            "all": all,
        }

        # 构建安全的import钩子，允许动态导入 re, math, datetime 常用验证包
        def safe_import(name, *args, **kwargs):
            if name in ["re", "math", "datetime", "json"]:
                return __import__(name, *args, **kwargs)
            raise ImportError(f"Import of '{name}' is strictly prohibited in sandbox.")

        # 放开 eval 沙箱安全限制，允许使用绝大多数安全内置函数和方法 (如 abs, keys, count 等)
        # 仅过滤可能导致危险行为或系统退出的底层内置函数
        safe_builtins = {k: v for k, v in builtins.__dict__.items() if k not in ["eval", "exec", "open", "__import__", "exit", "quit"]}
        safe_builtins["__import__"] = safe_import
        eval_globals = {"__builtins__": safe_builtins, **eval_env}
        
        for rule in code_rules:
            # 建立独立的局部命名空间，用于捕获多行脚本可能创建的 `is_valid` 变量
            eval_locals = {}
            try:
                # 尝试将其作为单纯可求值的单行表达式运行
                try:
                    is_valid = eval(rule, eval_globals, eval_locals)
                except SyntaxError:
                    # 如果语法错误（通常是因为多行赋值、循环等结构无法 eval），则回退走 exec
                    exec(rule, eval_globals, eval_locals)
                    # 从执行完毕的局部变量提取结果，默认未定义视为 False
                    is_valid = eval_locals.get("is_valid", False)
                    
                if not is_valid:
                    errors.append(f"验证未通过: {rule}")
            except SyntaxError:
                errors.append(f"规则 '{rule}' 不是有效的 Python 代码")
            except NameError as e:
                errors.append(f"规则 '{rule}' 存在未定义变量: {e}")
            except Exception as e:
                errors.append(f"评估规则 '{rule}' 时出错: {e}")

        return errors

    async def _run_llm_validation(
        self, fallback_rules: list[str], task: TaskRequest, result: TaskResult
    ) -> ReviewResult:
        """对无法代码化的规则，使用 LLM 进行语义审核。"""
        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个严格的质量审核员。请根据以下验证规则逐条审查任务结果数据。\n"
                    "如果所有规则均满足，返回 passed=true；否则返回 passed=false 并说明哪条规则未满足。\n"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"## 原始任务目标\n{task.goal}\n\n"
                    f"## 待验证的规则\n"
                    + "\n".join(f"- {r}" for r in fallback_rules)
                    + f"\n\n## 执行结果数据\n{json.dumps(result.data, ensure_ascii=False)}\n"
                ),
            },
        ]
        return await self._llm.chat(messages, response_model=ReviewResult)


# ── 向后兼容别名 ─────────────────────────────────────────────

HardcodeRuleReviewer = UnifiedValidator
LLMReviewer = UnifiedValidator


class CompositeReviewer(Reviewer):
    """组合审核器：Schema 校验 + 统一验收。"""

    def __init__(
        self,
        schema_reviewer: Reviewer | None = None,
        unified_validator: Reviewer | None = None,
        # 向后兼容旧参数名
        hardcode_reviewer: Reviewer | None = None,
        semantic_reviewer: Reviewer | None = None,
    ):
        self._schema = schema_reviewer or SchemaReviewer()
        # 优先使用新的 unified_validator，向后兼容旧的构造方式
        self._validator = unified_validator or UnifiedValidator()

    async def review(self, task: TaskRequest, result: TaskResult) -> ReviewResult:
        # 1. Schema 格式验证
        schema_res = await self._schema.review(task, result)
        if not schema_res.passed:
            return ReviewResult(passed=False, feedback=f"[Schema] {schema_res.feedback}")

        # 2. 统一验收（代码优先 + LLM 兜底）
        validator_res = await self._validator.review(task, result)
        if not validator_res.passed:
            return ReviewResult(passed=False, feedback=validator_res.feedback)

        return ReviewResult(
            passed=True,
            feedback=f"全部验证通过: [Schema] {schema_res.feedback} | [验收] {validator_res.feedback}",
        )
