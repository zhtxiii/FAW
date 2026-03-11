"""tests/test_models.py - 数据契约模型测试。"""

import pytest
from models import (
    ExploreResult,
    ReviewResult,
    RoutingDecision,
    SubTaskPlan,
    TaskRequest,
    TaskResult,
)


class TestRoutingDecision:
    def test_enum_values(self):
        assert RoutingDecision.SIMPLE.value == "SIMPLE"
        assert RoutingDecision.COMPLEX.value == "COMPLEX"
        assert RoutingDecision.UNKNOWN.value == "UNKNOWN"

    def test_enum_is_str(self):
        assert isinstance(RoutingDecision.SIMPLE, str)


class TestTaskRequest:
    def test_default_values(self):
        t = TaskRequest(goal="测试任务")
        assert t.goal == "测试任务"
        assert t.task_id  # 自动生成
        assert t.context == {}
        assert t.expected_output_schema == {}

    def test_custom_values(self):
        t = TaskRequest(
            task_id="t001",
            goal="自定义任务",
            context={"key": "value"},
            expected_output_schema={"type": "object"},
        )
        assert t.task_id == "t001"
        assert t.context["key"] == "value"

    def test_serialization_roundtrip(self):
        t = TaskRequest(task_id="t002", goal="序列化测试")
        json_str = t.model_dump_json()
        t2 = TaskRequest.model_validate_json(json_str)
        assert t == t2


class TestSubTaskPlan:
    def test_creation(self):
        plan = SubTaskPlan(
            sub_tasks=[
                TaskRequest(task_id="s1", goal="子任务1"),
                TaskRequest(task_id="s2", goal="子任务2"),
            ],
            reduction_logic="合并所有结果",
        )
        assert len(plan.sub_tasks) == 2
        assert plan.reduction_logic == "合并所有结果"

    def test_serialization(self):
        plan = SubTaskPlan(
            sub_tasks=[TaskRequest(task_id="s1", goal="子任务")],
            reduction_logic="直接使用",
        )
        data = plan.model_dump()
        plan2 = SubTaskPlan.model_validate(data)
        assert plan == plan2


class TestTaskResult:
    def test_success(self):
        r = TaskResult(task_id="t1", status="SUCCESS", data={"answer": 42})
        assert r.status == "SUCCESS"
        assert r.data["answer"] == 42
        assert r.artifacts == []

    def test_failed(self):
        r = TaskResult(task_id="t1", status="FAILED", data={"error": "timeout"})
        assert r.status == "FAILED"

    def test_invalid_status_rejected(self):
        with pytest.raises(Exception):
            TaskResult(task_id="t1", status="INVALID", data={})


class TestReviewResult:
    def test_passed(self):
        r = ReviewResult(passed=True, feedback="LGTM")
        assert r.passed is True
        assert r.feedback == "LGTM"

    def test_failed(self):
        r = ReviewResult(passed=False, feedback="数据格式错误")
        assert r.passed is False


class TestExploreResult:
    def test_with_context(self):
        e = ExploreResult(
            has_enough_context=True,
            new_context={"data": "found"},
            summary="发现了必要数据",
        )
        assert e.has_enough_context is True
        assert e.new_context["data"] == "found"

    def test_without_context(self):
        e = ExploreResult(
            has_enough_context=False,
            summary="信息不足",
        )
        assert e.has_enough_context is False
        assert e.new_context == {}
