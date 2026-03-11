"""FAW 数据契约模块。

使用 Pydantic v2 严格定义智能体之间通信的所有数据结构，
防止递归过程中的上下文崩溃。
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, List, Literal, Optional

from pydantic import BaseModel, Field


# ── 路由决策枚举 ─────────────────────────────────────────────

class RoutingDecision(str, Enum):
    """PlannerRouter 输出的路由决策。"""
    SIMPLE = "SIMPLE"    # 任务简单，直接执行
    COMPLEX = "COMPLEX"  # 任务复杂，需要 Map-Reduce 拆解
    UNKNOWN = "UNKNOWN"  # 缺乏信息，进入探索模式


# ── 任务输入契约 ─────────────────────────────────────────────

class TaskRequest(BaseModel):
    """智能体接收的任务描述。"""
    task_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    title: str = Field(default="", description="任务的短标题，用于可视化展示")
    goal: str
    context: dict[str, Any] = Field(default_factory=dict)
    expected_output_schema: dict[str, Any] = Field(default_factory=dict)
    depends_on: List[str] = Field(default_factory=list)
    hardcode_rules: List[str] = Field(default_factory=list, description="已废弃，请使用 validation_requirements")
    validation_requirements: List[str] = Field(default_factory=list, description="自然语言验证需求列表，由代码生成模型翻译为验证代码")
    strict_semantic_review: bool = Field(default=False, description="已废弃，统一验收器自动决定")


# ── Map 拆解契约 ─────────────────────────────────────────────

class SubTaskPlan(BaseModel):
    """主管智能体拆分任务时输出的拆解方案。"""
    sub_tasks: List[TaskRequest]
    reduction_logic: str


# ── 最终交付契约 ─────────────────────────────────────────────

class TaskResult(BaseModel):
    """智能体执行完毕后返回的结果。"""
    task_id: str
    status: Literal["SUCCESS", "FAILED"]
    data: Any = Field(
        default_factory=dict,
        description="必须严格符合 TaskRequest 中的 expected_output_schema",
    )
    artifacts: List[str] = Field(
        default_factory=list,
        description="产生的临时文件或代码路径",
    )

class SkillExecutionPlan(BaseModel):
    """SIMPLE 阶段的技能调用决策。规划器只需声明要用什么工具和参数。"""
    use_skill: bool
    skill_name: str = ""
    skill_params: Any = Field(default_factory=dict)
    direct_result: Any = Field(default_factory=dict)


class ExecutionPlannerResult(BaseModel):
    """超级调度器一次性给出的决策与配套方案"""
    decision: RoutingDecision
    simple_plan: Optional[SkillExecutionPlan] = None
    sub_tasks: Optional[List[TaskRequest]] = None
    reduction_logic: str = ""
    new_context_request: str = ""


# ── 技能合成契约 ─────────────────────────────────────────────

class SkillSynthesisRequest(BaseModel):
    """当请求的技能不存在时，触发开发智能体的请求体。"""
    skill_name: str
    skill_description: str = ""
    required_params: dict[str, Any] = Field(default_factory=dict)
    context: str = ""


class SkillSynthesisResult(BaseModel):
    """开发智能体交付的技能代码产物。"""
    skill_code: str
    class_name: str
    test_passed: bool = False
    error: str = ""


# ── 审核结果 ─────────────────────────────────────────────────

class ReviewResult(BaseModel):
    """审核守门员的输出。"""
    passed: bool
    feedback: str = ""


# ── 探索结果 ─────────────────────────────────────────────────

class ExploreResult(BaseModel):
    """探索模式的输出。"""
    has_enough_context: bool
    new_context: dict[str, Any] = Field(default_factory=dict)
    summary: str = ""
