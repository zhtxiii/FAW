"""tests/test_reviewer.py - 审核守门员测试。"""

import pytest
import pytest_asyncio

from models import ReviewResult, TaskRequest, TaskResult
from reviewer import (
    CompositeReviewer,
    UnifiedValidator,
    SchemaReviewer,
    Reviewer,
)


# ── SchemaReviewer 测试 ──────────────────────────────────────


class TestSchemaReviewer:
    @pytest.fixture
    def reviewer(self):
        return SchemaReviewer()

    @pytest.mark.asyncio
    async def test_no_schema_passes(self, reviewer):
        task = TaskRequest(goal="test", expected_output_schema={})
        result = TaskResult(task_id="t1", status="SUCCESS", data={"any": "data"})
        review = await reviewer.review(task, result)
        assert review.passed is True

    @pytest.mark.asyncio
    async def test_required_fields_present(self, reviewer):
        task = TaskRequest(
            goal="test",
            expected_output_schema={
                "required": ["name", "age"],
                "properties": {
                    "name": {"type": "string"},
                    "age": {"type": "integer"},
                },
            },
        )
        result = TaskResult(
            task_id="t1",
            status="SUCCESS",
            data={"name": "Alice", "age": 30},
        )
        review = await reviewer.review(task, result)
        assert review.passed is True

    @pytest.mark.asyncio
    async def test_required_fields_missing(self, reviewer):
        task = TaskRequest(
            goal="test",
            expected_output_schema={"required": ["name", "age"]},
        )
        result = TaskResult(
            task_id="t1", status="SUCCESS", data={"name": "Alice"}
        )
        review = await reviewer.review(task, result)
        assert review.passed is False
        assert "age" in review.feedback

    @pytest.mark.asyncio
    async def test_type_mismatch(self, reviewer):
        task = TaskRequest(
            goal="test",
            expected_output_schema={
                "properties": {
                    "count": {"type": "integer"},
                },
            },
        )
        result = TaskResult(
            task_id="t1", status="SUCCESS", data={"count": "not_a_number"}
        )
        review = await reviewer.review(task, result)
        assert review.passed is False
        assert "类型不匹配" in review.feedback

    @pytest.mark.asyncio
    async def test_various_types(self, reviewer):
        task = TaskRequest(
            goal="test",
            expected_output_schema={
                "properties": {
                    "s": {"type": "string"},
                    "n": {"type": "number"},
                    "b": {"type": "boolean"},
                    "a": {"type": "array"},
                    "o": {"type": "object"},
                },
            },
        )
        result = TaskResult(
            task_id="t1",
            status="SUCCESS",
            data={"s": "hello", "n": 3.14, "b": True, "a": [1, 2], "o": {"k": "v"}},
        )
        review = await reviewer.review(task, result)
        assert review.passed is True


# ── UnifiedValidator 测试 ────────────────────────────────────


class FakeLLMForValidator:
    """mock LLM 客户端，用于测试 UnifiedValidator。"""

    def __init__(self, code_rules=None, llm_fallback_rules=None, review_result=None):
        self._code_rules = code_rules or []
        self._llm_fallback_rules = llm_fallback_rules or []
        self._review_result = review_result or ReviewResult(passed=True, feedback="OK")

    async def chat_json(self, messages, temperature=0.3):
        return {
            "code_rules": self._code_rules,
            "llm_fallback_rules": self._llm_fallback_rules,
        }

    async def chat(self, messages, response_model=None, temperature=0.3):
        return self._review_result


class TestUnifiedValidator:

    @pytest.mark.asyncio
    async def test_no_requirements_passes(self):
        """无验证需求时自动通过。"""
        validator = UnifiedValidator()
        task = TaskRequest(goal="test")
        result = TaskResult(task_id="t1", status="SUCCESS", data={"x": 1})
        res = await validator.review(task, result)
        assert res.passed is True

    @pytest.mark.asyncio
    async def test_legacy_hardcode_rules_pass(self):
        """向后兼容：旧式 hardcode_rules（Python 代码）仍能正常执行。"""
        validator = UnifiedValidator()
        task = TaskRequest(
            goal="test",
            hardcode_rules=["data.get('age', 0) >= 18"],
        )
        result = TaskResult(task_id="t1", status="SUCCESS", data={"age": 20})
        res = await validator.review(task, result)
        assert res.passed is True

    @pytest.mark.asyncio
    async def test_legacy_hardcode_rules_fail(self):
        """向后兼容：旧式 hardcode_rules 不通过时返回失败。"""
        validator = UnifiedValidator()
        task = TaskRequest(
            goal="test",
            hardcode_rules=["data.get('age', 0) >= 18"],
        )
        result = TaskResult(task_id="t2", status="SUCCESS", data={"age": 16})
        res = await validator.review(task, result)
        assert res.passed is False
        assert "验证未通过" in res.feedback

    @pytest.mark.asyncio
    async def test_code_validation_from_natural_language(self):
        """自然语言需求被代码生成模型翻译为代码后验证通过。"""
        llm = FakeLLMForValidator(
            code_rules=["'$' in data.get('pricing', '')"],
            llm_fallback_rules=[],
        )
        validator = UnifiedValidator(llm=llm)
        task = TaskRequest(
            goal="test",
            validation_requirements=["必须包含美元价格"],
        )
        result = TaskResult(task_id="t1", status="SUCCESS", data={"pricing": "$20/月"})
        res = await validator.review(task, result)
        assert res.passed is True

    @pytest.mark.asyncio
    async def test_code_validation_fails(self):
        """代码验证不通过时直接拒绝（不浪费 LLM 调用）。"""
        llm = FakeLLMForValidator(
            code_rules=["'$' in data.get('pricing', '')"],
            llm_fallback_rules=[],
        )
        validator = UnifiedValidator(llm=llm)
        task = TaskRequest(
            goal="test",
            validation_requirements=["必须包含美元价格"],
        )
        result = TaskResult(task_id="t1", status="SUCCESS", data={"pricing": "免费"})
        res = await validator.review(task, result)
        assert res.passed is False

    @pytest.mark.asyncio
    async def test_llm_fallback_when_code_insufficient(self):
        """代码验证通过但存在无法代码化的规则时，LLM 兜底验证。"""
        llm = FakeLLMForValidator(
            code_rules=["len(data.get('items', [])) > 0"],
            llm_fallback_rules=["分析内容要有深度"],
            review_result=ReviewResult(passed=True, feedback="分析深入"),
        )
        validator = UnifiedValidator(llm=llm)
        task = TaskRequest(
            goal="test",
            validation_requirements=["列表不能为空", "分析内容要有深度"],
        )
        result = TaskResult(task_id="t1", status="SUCCESS", data={"items": [1, 2, 3]})
        res = await validator.review(task, result)
        assert res.passed is True

    @pytest.mark.asyncio
    async def test_llm_fallback_fails(self):
        """LLM 兜底验证不通过。"""
        llm = FakeLLMForValidator(
            code_rules=[],
            llm_fallback_rules=["分析内容要有深度"],
            review_result=ReviewResult(passed=False, feedback="内容过于肤浅"),
        )
        validator = UnifiedValidator(llm=llm)
        task = TaskRequest(
            goal="test",
            validation_requirements=["分析内容要有深度"],
        )
        result = TaskResult(task_id="t1", status="SUCCESS", data={"content": "很好"})
        res = await validator.review(task, result)
        assert res.passed is False
        assert "LLM" in res.feedback


# ── CompositeReviewer 测试 ───────────────────────────────────


class AlwaysPassReviewer(Reviewer):
    async def review(self, task, result):
        return ReviewResult(passed=True, feedback="PASSED")


class AlwaysFailReviewer(Reviewer):
    async def review(self, task, result):
        return ReviewResult(passed=False, feedback="FAILED")


class TestCompositeReviewer:

    @pytest.mark.asyncio
    async def test_all_pass(self):
        """所有子审核器通过则组合通过。"""
        composite = CompositeReviewer(
            schema_reviewer=AlwaysPassReviewer(),
            unified_validator=AlwaysPassReviewer(),
        )

        task = TaskRequest(goal="test")
        result = TaskResult(task_id="1", status="SUCCESS", data={})
        res = await composite.review(task, result)
        assert res.passed is True

    @pytest.mark.asyncio
    async def test_one_fails(self):
        """任一子审核器失败则组合失败。"""
        composite = CompositeReviewer(
            schema_reviewer=AlwaysPassReviewer(),
            unified_validator=AlwaysFailReviewer(),
        )

        task = TaskRequest(goal="test")
        result = TaskResult(task_id="1", status="SUCCESS", data={})
        res = await composite.review(task, result)
        assert res.passed is False
