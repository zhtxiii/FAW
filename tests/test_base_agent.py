"""tests/test_base_agent.py - BaseAgent 核心状态机测试（使用 mock LLM）。"""

import json
import pytest
from unittest.mock import AsyncMock, patch

from base_agent import BaseAgent, MaxRetryError
from models import (
    ExploreResult,
    ReviewResult,
    RoutingDecision,
    SubTaskPlan,
    TaskRequest,
    TaskResult,
)
from reviewer import SchemaReviewer
from skill_manager import SkillManager


class FakeLLMClient:
    """可编程的 mock LLM 客户端。"""

    def __init__(self):
        self._chat_responses = []
        self._chat_json_responses = []
        self._chat_call_count = 0
        self._chat_json_call_count = 0

    def set_chat_responses(self, responses):
        """设置 chat() 的返回序列。"""
        self._chat_responses = list(responses)
        self._chat_call_count = 0

    def set_chat_json_responses(self, responses):
        """设置 chat_json() 的返回序列。"""
        self._chat_json_responses = list(responses)
        self._chat_json_call_count = 0

    async def chat(self, messages, response_model=None, temperature=0.3):
        idx = self._chat_call_count
        self._chat_call_count += 1
        resp = self._chat_responses[idx % len(self._chat_responses)]

        if response_model is not None:
            # 当 response_model 是 ExecutionPlannerResult 且测试数据是路由字符串时，
            # 自动组装完整的 ExecutionPlannerResult 对象
            if response_model.__name__ == "ExecutionPlannerResult" and isinstance(resp, str):
                from models import ExecutionPlannerResult, RoutingDecision, SkillExecutionPlan, SubTaskPlan
                decision = RoutingDecision(resp)
                plan = ExecutionPlannerResult(decision=decision)

                if decision == RoutingDecision.SIMPLE:
                    json_resp = await self.chat_json(messages)
                    plan.simple_plan = SkillExecutionPlan.model_validate(json_resp)
                elif decision == RoutingDecision.COMPLEX:
                    # 从 chat_responses 中找下一个 SubTaskPlan 对象
                    for future_resp in self._chat_responses[idx + 1:]:
                        if isinstance(future_resp, SubTaskPlan):
                            plan.sub_tasks = future_resp.sub_tasks
                            plan.reduction_logic = future_resp.reduction_logic
                            break
                return plan

            if isinstance(resp, str):
                try:
                    return response_model.model_validate_json(resp)
                except Exception:
                    pass
            if isinstance(resp, response_model):
                return resp
            elif isinstance(resp, dict):
                return response_model.model_validate(resp)

        return resp

    async def chat_json(self, messages, temperature=0.3):
        idx = self._chat_json_call_count
        self._chat_json_call_count += 1
        return self._chat_json_responses[idx % len(self._chat_json_responses)]


class FakeReviewer:
    """可编程的 mock 审核器。"""

    def __init__(self, results: list[ReviewResult] | None = None):
        self._results = results or [ReviewResult(passed=True)]
        self._call_count = 0

    async def review(self, task, result):
        idx = self._call_count
        self._call_count += 1
        return self._results[idx % len(self._results)]


# ── SIMPLE 分支测试 ──────────────────────────────────────────


class TestBaseAgentSimple:
    @pytest.mark.asyncio
    async def test_simple_direct_result(self):
        """SIMPLE 分支：LLM 直接给出结果（不使用技能）。"""
        llm = FakeLLMClient()
        # 第 1 次 chat: 路由决策
        # chat_json: 分支 A 直接结果
        llm.set_chat_responses(["SIMPLE"])
        llm.set_chat_json_responses([{
            "use_skill": False,
            "direct_result": {"answer": 42},
        }])

        agent = BaseAgent(
            llm=llm,
            reviewer=FakeReviewer(),
        )

        task = TaskRequest(goal="返回 42")
        result = await agent.solve(task)

        assert result.status == "SUCCESS"
        assert result.data["answer"] == 42

    @pytest.mark.asyncio
    async def test_simple_with_skill(self):
        """SIMPLE 分支：LLM 选择使用计算器技能。"""
        llm = FakeLLMClient()
        llm.set_chat_responses(["SIMPLE"])
        llm.set_chat_json_responses([{
            "use_skill": True,
            "skill_name": "calculator",
            "skill_params": {"expression": "2 + 3"},
        }])

        agent = BaseAgent(
            llm=llm,
            reviewer=FakeReviewer(),
        )

        task = TaskRequest(goal="计算 2 + 3")
        result = await agent.solve(task)

        assert result.status == "SUCCESS"
        assert result.data["result"] == 5

    @pytest.mark.asyncio
    async def test_simple_skill_not_found_triggers_synthesis(self):
        """SIMPLE 分支：LLM 选择了不存在的技能，触发自动合成。"""
        llm = FakeLLMClient()
        llm.set_chat_responses(["SIMPLE"])
        llm.set_chat_json_responses([{
            "use_skill": True,
            "skill_name": "nonexistent_tool",
            "skill_params": {"x": 1},
        }])

        agent = BaseAgent(
            llm=llm,
            reviewer=FakeReviewer(),
        )

        # Mock synthesize_skill 返回 None（合成失败）
        async def mock_synthesize(*args, **kwargs):
            return None

        agent.skills.synthesize_skill = mock_synthesize

        task = TaskRequest(goal="测试")
        result = await agent.solve(task)
        assert result.status == "FAILED"
        assert "合成失败" in result.data["error"]

    @pytest.mark.asyncio
    async def test_simple_skill_synthesis_success(self):
        """SIMPLE 分支：技能不存在时自动合成成功并执行。"""
        from skill_manager import Skill

        class MockSynthesizedSkill(Skill):
            name = "synthesized_tool"
            description = "test"

            async def execute(self, params):
                return {"result": "synthesized_ok"}

        llm = FakeLLMClient()
        llm.set_chat_responses(["SIMPLE"])
        llm.set_chat_json_responses([{
            "use_skill": True,
            "skill_name": "synthesized_tool",
            "skill_params": {},
        }])

        agent = BaseAgent(
            llm=llm,
            reviewer=FakeReviewer(),
        )

        # Mock synthesize_skill 返回一个合成好的 Skill 实例
        mock_skill = MockSynthesizedSkill()

        async def mock_synthesize(*args, **kwargs):
            agent.skills.register(mock_skill)
            return mock_skill

        agent.skills.synthesize_skill = mock_synthesize

        task = TaskRequest(goal="测试合成技能")
        result = await agent.solve(task)
        assert result.status == "SUCCESS"
        assert result.data["result"] == "synthesized_ok"


# ── COMPLEX 分支测试 ─────────────────────────────────────────


class TestBaseAgentComplex:
    @pytest.mark.asyncio
    async def test_complex_map_reduce(self):
        """COMPLEX 分支：完整的 Map-Reduce 流程。"""
        llm = FakeLLMClient()

        # 调用序列：
        # 1. 根 agent 路由: COMPLEX
        # 2. 根 agent Map 拆解: SubTaskPlan
        # 3. 子 agent 1 路由: SIMPLE
        # 4. 子 agent 2 路由: SIMPLE
        # 最终 Reduce 汇总
        llm.set_chat_responses([
            "COMPLEX",                       # 根路由
            SubTaskPlan(                     # Map 拆解
                sub_tasks=[
                    TaskRequest(task_id="s1", goal="子任务1"),
                    TaskRequest(task_id="s2", goal="子任务2"),
                ],
                reduction_logic="合并结果",
            ),
            "SIMPLE",                        # 子 agent 1 路由
            "SIMPLE",                        # 子 agent 2 路由
        ])

        llm.set_chat_json_responses([
            {"use_skill": False, "direct_result": {"part": "A"}},   # 子 agent 1 结果
            {"use_skill": False, "direct_result": {"part": "B"}},   # 子 agent 2 结果
            {"merged": "A+B"},                                      # Reduce 汇总
        ])

        agent = BaseAgent(
            llm=llm,
            reviewer=FakeReviewer(),
        )

        task = TaskRequest(goal="复杂任务")
        result = await agent.solve(task)
        assert result.status == "SUCCESS"

    @pytest.mark.asyncio
    async def test_depth_limit_forces_simple(self):
        """达到最大深度时，COMPLEX 应强制降级为 SIMPLE。"""
        llm = FakeLLMClient()
        llm.set_chat_responses(["COMPLEX"])  # 路由说是 COMPLEX
        llm.set_chat_json_responses([{
            "use_skill": False,
            "direct_result": {"fallback": True},
        }])

        agent = BaseAgent(
            llm=llm,
            reviewer=FakeReviewer(),
            max_depth=3,
            current_depth=3,  # 已达到最大深度
        )

        task = TaskRequest(goal="应该被降级的复杂任务")
        result = await agent.solve(task)
        assert result.status == "SUCCESS"
        assert "Depth" in result.data["error"]


# ── 审核重试测试 ─────────────────────────────────────────────


class TestBaseAgentReview:
    @pytest.mark.asyncio
    async def test_review_retry_then_pass(self):
        """审核第一次失败，第二次通过。"""
        llm = FakeLLMClient()
        llm.set_chat_responses(["SIMPLE", "SIMPLE"])
        llm.set_chat_json_responses([
            {"use_skill": False, "direct_result": {"v": 1}},
            {"use_skill": False, "direct_result": {"v": 2}},
        ])

        reviewer = FakeReviewer([
            ReviewResult(passed=False, feedback="请修正"),
            ReviewResult(passed=True),
        ])

        agent = BaseAgent(llm=llm, reviewer=reviewer)
        task = TaskRequest(goal="需要重试的任务")
        result = await agent.solve(task)
        assert result.status == "SUCCESS"

    @pytest.mark.asyncio
    async def test_max_retry_exceeded(self):
        """超过最大重试次数应抛出 MaxRetryError。"""
        llm = FakeLLMClient()
        # 需要 1 次路由 + MAX_RETRIES+1 次执行
        llm.set_chat_responses(["SIMPLE"] * 10)
        llm.set_chat_json_responses([
            {"use_skill": False, "direct_result": {"bad": True}}
        ] * 10)

        reviewer = FakeReviewer([
            ReviewResult(passed=False, feedback="一直不通过"),
        ])

        agent = BaseAgent(llm=llm, reviewer=reviewer)
        task = TaskRequest(goal="永远过不了审核的任务")

        with pytest.raises(MaxRetryError) as exc_info:
            await agent.solve(task)
        assert exc_info.value.task_id == task.task_id


# ── UNKNOWN 分支测试 ─────────────────────────────────────────


class TestBaseAgentUnknown:
    @pytest.mark.asyncio
    async def test_explore_then_solve(self):
        """UNKNOWN 分支：探索成功后重新路由为 SIMPLE 并完成。"""
        llm = FakeLLMClient()
        llm.set_chat_responses([
            "UNKNOWN",                 # 第一次路由
            ExploreResult(             # 探索结果：成功
                has_enough_context=True,
                new_context={"extra": "info"},
                summary="找到了必要信息",
            ),
            "SIMPLE",                  # 第二次路由（重入 solve）
        ])
        llm.set_chat_json_responses([{
            "use_skill": False,
            "direct_result": {"solved": True},
        }])

        agent = BaseAgent(llm=llm, reviewer=FakeReviewer(), skills=SkillManager())
        task = TaskRequest(goal="未知任务")
        result = await agent.solve(task)
        assert result.status == "SUCCESS"
        assert result.data["solved"] is True

    @pytest.mark.asyncio
    async def test_explore_exhausted(self):
        """UNKNOWN 分支：探索 3 次仍无进展，返回 FAILED。"""
        llm = FakeLLMClient()
        llm.set_chat_responses([
            "UNKNOWN",
            ExploreResult(has_enough_context=False, summary="信息不足1"),
            ExploreResult(has_enough_context=False, summary="信息不足2"),
            ExploreResult(has_enough_context=False, summary="信息不足3"),
        ])

        # ask_human 返回 FAILED，此结果会经过 reviewer
        # 使用一个总是通过的 reviewer
        agent = BaseAgent(llm=llm, reviewer=FakeReviewer(), skills=SkillManager())
        task = TaskRequest(goal="无法探索的任务")
        result = await agent.solve(task)
        assert result.status == "FAILED"
        assert "error" in result.data
