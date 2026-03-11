"""tests/test_planner_router.py - 决策路由引擎测试（使用 mock LLM）。"""

import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from models import RoutingDecision, TaskRequest
from planner_router import PlannerRouter


class MockLLMClient:
    """模拟 LLM 客户端，返回预设的路由决策字符串。"""

    def __init__(self, response: str):
        self._response = response
        self.last_messages = None # To store messages for inspection

    async def chat(self, messages, response_model=None, temperature=0.3):
        self.last_messages = messages # Store messages
        return self._response


class TestPlannerRouter:
    @pytest.mark.asyncio
    async def test_route_simple(self):
        llm = MockLLMClient("SIMPLE")
        router = PlannerRouter(llm=llm)
        task = TaskRequest(goal="简单计算 1+1")
        decision = await router.evaluate(task)
        assert decision == RoutingDecision.SIMPLE

    @pytest.mark.asyncio
    async def test_route_complex(self):
        llm = MockLLMClient("COMPLEX")
        router = PlannerRouter(llm=llm)
        task = TaskRequest(goal="分析多个数据集并交叉对比")
        decision = await router.evaluate(task)
        assert decision == RoutingDecision.COMPLEX

    @pytest.mark.asyncio
    async def test_route_unknown(self):
        llm = MockLLMClient("UNKNOWN")
        router = PlannerRouter(llm=llm)
        task = TaskRequest(goal="完全不确定的任务")
        decision = await router.evaluate(task)
        assert decision == RoutingDecision.UNKNOWN

    @pytest.mark.asyncio
    async def test_route_fallback_on_invalid(self):
        """无法解析的 LLM 输出应回退到 UNKNOWN。"""
        llm = MockLLMClient("我不知道这是什么")
        router = PlannerRouter(llm=llm)
        task = TaskRequest(goal="测试任务")
        decision = await router.evaluate(task)
        assert decision == RoutingDecision.UNKNOWN

    @pytest.mark.asyncio
    async def test_route_with_trailing_text(self):
        """LLM 输出包含多余文字时仍能正确解析。"""
        llm = MockLLMClient("我认为这是 SIMPLE 的任务")
        router = PlannerRouter(llm=llm)
        task = TaskRequest(goal="简单任务")
        decision = await router.evaluate(task)
        assert decision == RoutingDecision.SIMPLE

    @pytest.mark.asyncio
    async def test_route_with_tools_info(self):
        """验证技能信息能正确传递给 LLM。"""
        llm = MockLLMClient("SIMPLE")
        tools = [{"name": "calculator", "description": "计算器"}]
        router = PlannerRouter(llm=llm, available_skills=tools)
        task = TaskRequest(goal="计算数学表达式")
        decision = await router.evaluate(task)
        assert decision == RoutingDecision.SIMPLE
