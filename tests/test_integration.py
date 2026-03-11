"""tests/test_integration.py - 端到端集成测试。"""

import pytest

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
from skill_manager import create_default_registry


class StepLLMClient:
    """按步骤返回预定义响应的 mock LLM 客户端。

    由于加入了 asyncio 并发调度，严格依赖索引顺序可能会出错，
    因此改为基于消息内容的一些关键词进行路由。
    """

    def __init__(self, chat_steps: list, json_steps: list):
        self._chat_steps = chat_steps
        self._json_steps = json_steps
        self._chat_idx = 0
        self._json_idx = 0

    async def chat(self, messages, response_model=None, temperature=0.3):
        content = " ".join(str(m.get("content", "")) for m in messages)
        
        # 1. 探索模式
        if response_model and response_model.__name__ == "ExploreResult":
            resp = [s for s in self._chat_steps if type(s).__name__ == "ExploreResult"][0]
            # 为了多步测试保持连贯
            for step in self._chat_steps:
               if type(step).__name__ == "ExploreResult":
                   if "has_enough_context=False" in str(step) and "信息不足" in str(step.summary) and str(self._chat_idx) in step.summary:
                       resp = step
            return resp
            
        # 2. 规划评估阶段
        if "战略先锋节点" in content:
            # 找到下一个字符串标志 (SIMPLE, COMPLEX, UNKNOWN)
            resp_str = "SIMPLE" # 默认 fallback
            while self._chat_idx < len(self._chat_steps):
                item = self._chat_steps[self._chat_idx]
                self._chat_idx += 1
                if isinstance(item, str):
                    resp_str = item
                    break
            
            from models import ExecutionPlannerResult, RoutingDecision, SkillExecutionPlan
            plan = ExecutionPlannerResult(decision=RoutingDecision(resp_str))
            
            if plan.decision == RoutingDecision.SIMPLE:
                json_resp = await self.chat_json(messages)
                plan.simple_plan = SkillExecutionPlan.model_validate(json_resp)
            elif plan.decision == RoutingDecision.COMPLEX:
                sub = next((s for s in self._chat_steps if type(s).__name__ == "SubTaskPlan"), None)
                if sub:
                    plan.sub_tasks = sub.sub_tasks
                    plan.reduction_logic = sub.reduction_logic
            return plan

        # 默认备用
        if self._chat_idx < len(self._chat_steps):
             resp = self._chat_steps[self._chat_idx]
             self._chat_idx += 1
             return resp
        return "SIMPLE"

    async def chat_json(self, messages, temperature=0.3):
        content = " ".join(str(m.get("content", "")) for m in messages)
        import json
        
        # 汇总阶段
        if "缝合梭机" in content:
            return next((s for s in self._json_steps if "use_skill" not in s and "direct_result" not in s), {"merged": "OK"})
            
        # SIMPLE 阶段: 寻找包含了相关关键词的 mock 返回
        if "计算 2+3" in content:
            return next((s for s in self._json_steps if "use_skill" in s and "2+3" in json.dumps(s)), self._json_steps[0])
        elif "计算 4+5" in content:
            return next((s for s in self._json_steps if "use_skill" in s and "4+5" in json.dumps(s)), self._json_steps[1])
            
        resp = self._json_steps[self._json_idx % len(self._json_steps)]
        self._json_idx += 1
        return resp


class AlwaysPassReviewer:
    """总是通过的审核器。"""

    async def review(self, task, result):
        return ReviewResult(passed=True, feedback="OK")


class TestIntegration:
    @pytest.mark.asyncio
    async def test_simple_end_to_end(self):
        """完整的 SIMPLE 分支端到端流程。"""
        llm = StepLLMClient(
            chat_steps=["SIMPLE"],
            json_steps=[{
                "use_skill": True,
                "skill_name": "calculator",
                "skill_params": {"expression": "10 * 5 + 2"},
            }],
        )

        agent = BaseAgent(llm=llm, reviewer=AlwaysPassReviewer())
        task = TaskRequest(
            goal="计算 10 * 5 + 2",
            expected_output_schema={
                "properties": {"result": {"type": "number"}},
            },
        )
        result = await agent.solve(task)
        assert result.status == "SUCCESS"
        assert result.data["result"] == 52

    @pytest.mark.asyncio
    async def test_complex_end_to_end(self):
        """完整的 COMPLEX Map-Reduce 端到端流程。"""
        llm = StepLLMClient(
            chat_steps=[
                "COMPLEX",                          # 根路由
                SubTaskPlan(                        # Map 拆解
                    sub_tasks=[
                        TaskRequest(task_id="sub1", goal="计算 2+3"),
                        TaskRequest(task_id="sub2", goal="计算 4+5"),
                    ],
                    reduction_logic="将两个计算结果相加得到最终答案",
                ),
                "SIMPLE",                           # 子 1 路由
                "SIMPLE",                           # 子 2 路由
            ],
            json_steps=[
                {"use_skill": True, "skill_name": "calculator", "skill_params": {"expression": "2+3"}},
                {"use_skill": True, "skill_name": "calculator", "skill_params": {"expression": "4+5"}},
                {"total": 14},                      # Reduce 汇总
            ],
        )

        agent = BaseAgent(llm=llm, reviewer=AlwaysPassReviewer())
        task = TaskRequest(goal="分步计算 (2+3)+(4+5)")
        result = await agent.solve(task)
        assert result.status == "SUCCESS"
        assert result.data["total"] == 14

    @pytest.mark.asyncio
    async def test_unknown_to_simple_end_to_end(self):
        """UNKNOWN 分支探索成功后转 SIMPLE 完成。"""
        llm = StepLLMClient(
            chat_steps=[
                "UNKNOWN",                         # 第一次路由
                ExploreResult(                     # 探索结果
                    has_enough_context=True,
                    new_context={"formula": "6*7"},
                    summary="找到了计算公式",
                ),
                "SIMPLE",                          # 重新路由
            ],
            json_steps=[{
                "use_skill": True,
                "skill_name": "calculator",
                "skill_params": {"expression": "6*7"},
            }],
        )

        manager = create_default_registry()
        if "tavily_search" in manager._skills:
            del manager._skills["tavily_search"]
        agent = BaseAgent(llm=llm, reviewer=AlwaysPassReviewer(), skills=manager)
        task = TaskRequest(goal="计算某个未知公式的结果")
        result = await agent.solve(task)
        assert result.status == "SUCCESS"
        assert result.data["result"] == 42

    @pytest.mark.asyncio
    async def test_schema_review_integration(self):
        """使用真实的 SchemaReviewer 测试 schema 验证集成。"""
        llm = StepLLMClient(
            chat_steps=["SIMPLE"],
            json_steps=[{
                "use_skill": False,
                "direct_result": {"name": "FAW", "version": 1},
            }],
        )

        agent = BaseAgent(llm=llm, reviewer=SchemaReviewer())
        task = TaskRequest(
            goal="返回项目信息",
            expected_output_schema={
                "required": ["name", "version"],
                "properties": {
                    "name": {"type": "string"},
                    "version": {"type": "integer"},
                },
            },
        )
        result = await agent.solve(task)
        assert result.status == "SUCCESS"
        assert result.data["name"] == "FAW"

    @pytest.mark.asyncio
    async def test_depth_limit_integration(self):
        """深度限制防死锁集成测试。"""
        llm = StepLLMClient(
            chat_steps=["COMPLEX"],  # 想走 COMPLEX 但深度已满
            json_steps=[{
                "use_skill": False,
                "direct_result": {"degraded": True},
            }],
        )

        agent = BaseAgent(
            llm=llm,
            reviewer=AlwaysPassReviewer(),
            max_depth=2,
            current_depth=2,
        )
        task = TaskRequest(goal="应被降级处理的任务")
        result = await agent.solve(task)
        assert result.status == "SUCCESS"
        assert "error" in result.data
        assert "Depth" in result.data["error"]
