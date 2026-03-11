"""FAW 执行节点基类。

实现完整的核心状态机：
  solve(task) ->
    Step 1: One-Shot 规划大脑生成执行策略与决策
    Step 2: 根据决策调度 (SIMPLE / COMPLEX / UNKNOWN) 子分支
    Step 3: Reviewer 审核与携带 Feedback 重试循坏
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from typing import Any

from config import DEFAULT_MAX_DEPTH, MAX_EXPLORE_RETRIES, MAX_RETRIES
from llm_client import LLMClient
from models import (
    ExploreResult,
    RoutingDecision,
    SubTaskPlan,
    TaskRequest,
    TaskResult,
    ExecutionPlannerResult,
    SkillExecutionPlan,
)
from reviewer import CompositeReviewer, UnifiedValidator, Reviewer, SchemaReviewer
from skill_manager import SkillManager, create_default_registry

logger = logging.getLogger(__name__)
tasks_logger = logging.getLogger("[TASKS]")


class MaxRetryError(Exception):
    """超出最大重试次数时抛出。"""

    def __init__(self, task_id: str, retries: int, last_feedback: str):
        self.task_id = task_id
        self.retries = retries
        self.last_feedback = last_feedback
        super().__init__(
            f"Task {task_id} failed after {retries} retries. Last feedback: {last_feedback}"
        )


class DepthExceededError(Exception):
    """超出最大递归深度时抛出（强制降级也失败的情况）。"""
    pass


class BaseAgent:
    """执行节点基类。所有智能体（不管是顶层还是第 N 层）都基于此类。"""

    def __init__(
        self,
        role: str = "通用智能体",
        skills: SkillManager | None = None,
        max_depth: int = DEFAULT_MAX_DEPTH,
        current_depth: int = 0,
        llm: LLMClient | None = None,
        reviewer: Reviewer | None = None,
        agent_id: str | None = None,
    ):
        self.agent_id = agent_id or uuid.uuid4().hex[:8]
        self.role = role
        self.skills = skills or create_default_registry()
        self.max_depth = max_depth
        self.current_depth = current_depth
        self._llm = llm or LLMClient()
        if reviewer:
            self._reviewer = reviewer
        else:
            self._reviewer = CompositeReviewer(
                schema_reviewer=SchemaReviewer(),
                unified_validator=UnifiedValidator(self._llm),
            )

    def _log(self, msg: str, *args: Any) -> None:
        prefix = f"[Agent {self.agent_id} depth={self.current_depth}]"
        logger.info(f"{prefix} {msg}", *args)

    # ── 核心入口 ─────────────────────────────────────────────

    async def solve(self, task: TaskRequest) -> TaskResult:
        """核心状态机入口。"""
        # 为节点赋予人类可读的名牌（如果有）
        if task.title:
            self.agent_id = task.title

        self._log("Received task: %s", task.goal)

        retry_count = 0
        feedback_context = ""

        # 全局大重试循环：如果有拦截或失效，将拿回 Feedback 重头做规划
        while retry_count <= MAX_RETRIES:
            # Step 1: 超级规划引擎（一次性下发路由决策与相应配置详情）
            plan = await self._analyze_and_plan(task, feedback_context)

            # Step 2: 方案推演
            temp_result = await self._execute_plan(task, plan)

            # Step 3: 原教旨审核
            review = await self._reviewer.review(task, temp_result)

            if review.passed:
                self._log("Task %s passed review.", task.task_id)
                return temp_result

            # 审核未通过：积淀纠错基因并回旋镖重头再来
            retry_count += 1
            feedback_context = review.feedback
            self._log(
                "Task %s review failed (retry %d/%d): %s",
                task.task_id,
                retry_count,
                MAX_RETRIES,
                review.feedback,
            )

            if retry_count > MAX_RETRIES:
                raise MaxRetryError(task.task_id, MAX_RETRIES, review.feedback)

        raise MaxRetryError(task.task_id, MAX_RETRIES, feedback_context)

    # ── Step 1: 规划合并网络 ─────────────────────────────────

    async def _analyze_and_plan(self, task: TaskRequest, feedback: str = "") -> ExecutionPlannerResult:
        """核心大脑：一次性完成任务路况判定与方案生成"""
        self._log("Analyzing context and generating integrated operational plan...")
        skills_info = self.skills.list_skills()
        
        feedback_hint = ""
        if feedback:
            feedback_hint = f"\n\n🚨 注意：上一次生成的策略执行被安全审计打回。反省反馈：{feedback}。请必须纠正方案！"

        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个掌控全局调度的战略先锋节点。你需要审视最终任务目标、约束上下文以及技能兵器库，"
                    "选择一条最佳的拓扑进攻分支（SIMPLE即直降落, COMPLEX即裂变编队, UNKNOWN即打探虚实）。\n"
                    "一切产物必须遵循包含决策字段的高阶复合结构 `ExecutionPlannerResult`，请输出纯 JSON!\n\n"
                    "请按以下指导方针严格填充 JSON 数据格式：\n"
                    "- COMPLEX（复杂战区）：当前目标太过宏大，或所需操作高度依赖前后串行异步时使用。\n"
                    "  > 必须完整填充 `sub_tasks`（任务细分拓扑列表）与 `reduction_logic`（将战果缝合的逻辑陈述）。\n"
                    "  > 切记为每一个子节点亲自指派独一无二的名称至 `task_id`，并使用 `depends_on` 挂钩其必须等待的关联 `task_id`！为子任务赋予一句便于人类视网膜读取的短语填入 `title` （比如 'search_ai_news'）。并提供清晰的预期 schema 结构。\n"
                    "  > 约束字段 `validation_requirements` 必须是自然语言描述的验证需求，例如：['返回结果必须包含具体的美元价格数字', '优点列表不得为空']。你不需要编写任何代码，系统会自动将这些需求翻译为验证代码。\n"
                    "- SIMPLE（简单单发）：只要调用一条 API 知识请求，或者手搓一段数据处理逻辑就能立竿见影的简单交付。\n"
                    "  > 必须填写 `simple_plan`，指定要调用的 Skill 名称（`skill_name`）与参数（`skill_params`）。\n"
                    "  > 🚨重要准则🚨：在选择复用现有的系统 Skill 时，**必须严格比对**该 Skill 描述支持的输出格式与当前子任务的 `expected_output_schema`。\n"
                    "  > 绝大多数简单原子工具（如基础文本处理）只能返回固定格式（如 `{'result': '...'}`），它们**绝对无法**满足复杂的定制结构提取要求（如 `{'selected_news': '...', 'selection_reason': '...'}`）。\n"
                    "  > 如果现有工具的输出能力与 Schema 不吻合，你**必须**声明一个**全新且不存在**的工具名称，系统会自动按照复杂的输出 Schema 现场合成对应的定制工具代码。千万不要强行让简单工具去完成复杂 Schema 提取任务。\n"
                    "  > 另外，如果任务是纯语言分析，你也可以选择直接由大模型给出答案，这时设置 `use_skill: false`，并将符合 expected_output_schema 格式的答案直接放入 `direct_result` 字典中。\n"
                    "- UNKNOWN（绝境试探）：任务根本读不懂，缺斤少两无从下手。\n"
                    "  > 填写 `new_context_request` 指定你想要外挂的上下文补给。\n\n"
                    "【严格输出约束】\n"
                    "1. 你的输出必须是且仅是一个合法的 JSON 对象，不得包含任何额外文字、注释或 markdown 标记。\n"
                    "2. 子任务的 expected_output_schema 中声明的 required 字段，执行此子任务的载荷必须输出与其严格对应的字段名以及数据类型。\n"
                    "3. 如果 `use_skill` 设置为 true 并且调用的是**已有**技能，请极度小心它是否能输出所需的 Schema！如果不行，请编造一个新的技能名称。\n"
                    "4. 禁止定义过于苛刻或不可控的 validation_requirements，验证需求应当可以被执行结果合理满足。\n"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"## 标靶\n- {task.goal}\n"
                    f"## 战区脉络 (Context)\n{json.dumps(task.context, ensure_ascii=False)}\n"
                    f"## 收网格式 (Schema)\n{json.dumps(task.expected_output_schema, ensure_ascii=False)}\n\n"
                    f"## 可用工具库 (Skills)\n{json.dumps(skills_info, ensure_ascii=False)}"
                    f"{feedback_hint}"
                ),
            },
        ]

        plan = await self._llm.chat(messages, response_model=ExecutionPlannerResult)

        # 全局递归雪球深度熔断底座（强制退守单兵模式）
        if plan.decision == RoutingDecision.COMPLEX and self.current_depth >= self.max_depth:
            self._log(
                "Recursion overload! Deep %d reaches configured maximum %d. Enforcing COMPLEX to SIMPLE downgrade.",
                self.current_depth,
                self.max_depth,
            )
            # 既然无路可退，强制大模型直接根据现有的 Context 强行生成最终 Schema 数据！
            schema_str = json.dumps(task.expected_output_schema, ensure_ascii=False) if task.expected_output_schema else "{}"
            context_str = json.dumps(task.context, ensure_ascii=False)
            fallback_messages = [
                {
                    "role": "system",
                    "content": (
                        "你的上级调度器已经耗尽了递归资源，你被强制要求立刻终止拆分，就地总结！\n"
                        "不管当前拥有的信息是否完备，你必须基于【任务上下文】强行拼凑/推理出复合【目标 Schema】要求的答案。\n"
                        "仅返回合法的 JSON 对象，不要做任何解释。"
                    )
                },
                {
                    "role": "user",
                    "content": f"【任务目标】: {task.goal}\n【任务上下文】: {context_str}\n【期望的输出 Schema】: {schema_str}"
                }
            ]
            
            try:
                forced_result = await self._llm.chat_json(fallback_messages)
            except Exception as e:
                self._log("Fallback LLM synthesis failed: %s", e)
                forced_result = {"error": "Depth threshold triggered, forced termination."}

            plan.decision = RoutingDecision.SIMPLE
            plan.simple_plan = SkillExecutionPlan(
                use_skill=False, 
                direct_result=forced_result
            )
            
        return plan

    # ── Step 2: 方案委派 ────────────────────────────────────

    async def _execute_plan(self, task: TaskRequest, plan: ExecutionPlannerResult) -> TaskResult:
        """兵分三路推演生成的战略蓝本。"""
        self._log("Selected strategic route: %s", plan.decision.value)

        if plan.decision == RoutingDecision.SIMPLE:
            if not plan.simple_plan:
                return TaskResult(task_id=task.task_id, status="FAILED", data={"error": "SIMPLE decision without plan"})
            return await self._execute_simple(task, plan.simple_plan)
            
        elif plan.decision == RoutingDecision.COMPLEX:
            if not plan.sub_tasks:
                return TaskResult(task_id=task.task_id, status="FAILED", data={"error": "COMPLEX decision without sub_tasks"})
            return await self._execute_complex(task, plan)
            
        elif plan.decision == RoutingDecision.UNKNOWN:
            return await self._branch_unknown(task)
            
        return TaskResult(task_id=task.task_id, status="FAILED", data={"error": f"Unknown decision enum: {plan.decision}"})

    # ── 分支 A: SIMPLE 单兵执行 ─────────────────────────────

    async def _execute_simple(self, task: TaskRequest, plan: SkillExecutionPlan) -> TaskResult:
        """单刀直入调用底层载荷弹药包或直给回答"""
        if plan.use_skill and plan.skill_name:
            skill = self.skills.get(plan.skill_name)

            # 工具库中不存在该技能 -> 触发开发智能体异步合成
            if not skill:
                self._log(
                    "Skill '%s' not found in registry, triggering synthesis...",
                    plan.skill_name,
                )
                skill = await self.skills.synthesize_skill(
                    skill_name=plan.skill_name,
                    skill_params=plan.skill_params if isinstance(plan.skill_params, dict) else {},
                    task_goal=task.goal,
                    expected_output_schema=task.expected_output_schema,
                    llm=self._llm,
                )
                if not skill:
                    return TaskResult(
                        task_id=task.task_id,
                        status="FAILED",
                        data={"error": f"技能 '{plan.skill_name}' 不存在且自动合成失败"},
                    )
            
            self._log("Deploying payload: %s | Params: %s", plan.skill_name, plan.skill_params)
            try:
                data = await skill.execute(plan.skill_params)
            except Exception as e:
                self._log("Payload deployment collapsed: %s", e)
                return TaskResult(task_id=task.task_id, status="FAILED", data={"error": str(e)})

            # ── 智能体理解与 Schema 适配层 ──
            if task.expected_output_schema:
                self._log("Invoking LLM comprehension layer to map output to Schema...")
                
                schema_str = json.dumps(task.expected_output_schema, ensure_ascii=False)
                data_str = json.dumps(data, ensure_ascii=False)
                
                messages = [
                    {
                        "role": "system",
                        "content": (
                            "你是一个执行助手。你刚刚调用了一个底层逻辑工具来完成任务。\n"
                            "你的职责是：阅读工具返回的原始生数据，理解它的内涵，并从中提取/总结需要的信息，"
                            "最后精准地组装并填充到下方的预期 JSON 格式规范 (Schema) 中返还。\n\n"
                            "【严格约束】：\n"
                            "1. 你必须且只能输出合法的 JSON 对象。\n"
                            "2. 如果原始数据缺乏足够信息填满所有 required 字段，请运用合理推断、填充默认值或加注解释说明。\n"
                            "3. 绝对不要输出 markdown 块，不要写任何前言后语废话。"
                        )
                    },
                    {
                        "role": "user",
                        "content": (
                            f"【当前子任务目标】\n{task.goal}\n\n"
                            f"【底层工具返回的原始数据】\n{data_str}\n\n"
                            f"【请将结果装入以下目标 Schema】\n{schema_str}"
                        )
                    }
                ]
                
                try:
                    adapted_data = await self._llm.chat_json(messages)
                    data = adapted_data
                    self._log("LLM Schema comprehension successful.")
                except Exception as e:
                    self._log("LLM comprehension layer collapsed: %s", e)
                    return TaskResult(task_id=task.task_id, status="FAILED", data={"error": f"Schema适配理解层失败: {str(e)}"})
        else:
            # 大脑直接推理而得的裸结果
            data = plan.direct_result

        return TaskResult(task_id=task.task_id, status="SUCCESS", data=data)

    # ── 分支 B: COMPLEX 编队爆破 ────────────────────────────

    async def _execute_complex(self, task: TaskRequest, plan: ExecutionPlannerResult) -> TaskResult:
        """复杂的战术地图多点布控与收束整合"""
        sub_tasks = plan.sub_tasks or []
        self._log("Orchestrating %d multi-vector fronts...", len(sub_tasks))
        
        results = await self._dispatch(sub_tasks)
        return await self._reduce(task, results, plan.reduction_logic)

    async def _dispatch(self, sub_tasks: list[TaskRequest]) -> list[TaskResult]:
        """按网络拓扑顺延依赖分发集群调度"""
        results_map: dict[str, TaskResult] = {}
        pending_deps: dict[str, set[str]] = {
            task.task_id: set(task.depends_on) for task in sub_tasks
        }
        task_map: dict[str, TaskRequest] = {task.task_id: task for task in sub_tasks}
        
        dependents: dict[str, list[str]] = {task.task_id: [] for task in sub_tasks}
        for task in sub_tasks:
            for dep in task.depends_on:
                if dep in dependents:
                    dependents[dep].append(task.task_id)
                else:
                    logger.warning("Agent 骨架断裂: 任务 %s 宣称必须依赖幽灵任务 %s，将自动削减。规则引擎已规避。", task.task_id, dep)
                    pending_deps[task.task_id].discard(dep)

        ready_queue: asyncio.Queue[TaskRequest] = asyncio.Queue()
        for task_id, deps in pending_deps.items():
            if not deps:
                ready_queue.put_nowait(task_map[task_id])

        all_completed_event = asyncio.Event()
        if not sub_tasks:
            all_completed_event.set()
        completed_count = 0
        running_count = 0
        total_count = len(sub_tasks)

        async def worker():
            import config
            nonlocal completed_count, running_count
            while completed_count < total_count:
                try:
                    task = await asyncio.wait_for(ready_queue.get(), timeout=1.0)
                    running_count += 1
                except asyncio.TimeoutError:
                    if completed_count < total_count and ready_queue.empty() and running_count == 0:
                        stuck = [tid for tid, deps in pending_deps.items() if deps and tid not in results_map]
                        logger.error("战吼系统阻滞：侦测到底层闭环死锁，强行中断死缠烂打的幽灵 %s", stuck)
                        for tid in stuck:
                            results_map[tid] = TaskResult(
                                task_id=tid, status="FAILED", data={"error": "依赖关系死锁陷阱"}
                            )
                            completed_count += 1
                        all_completed_event.set()
                        break
                    continue

                if config.DEBUG_TASKS:
                    # 使用 title 以提升交互可视化美感
                    display_name = task.title if task.title else task.task_id
                    tasks_logger.debug(f"▶️ [子任务执行] 正在启动 {display_name} (目标: {task.goal})")

                failed_deps_id = [
                    dep_id for dep_id in task.depends_on 
                    if dep_id not in results_map or results_map[dep_id].status != "SUCCESS"
                ]
                
                if failed_deps_id:
                    result = TaskResult(
                        task_id=task.task_id,
                        status="FAILED",
                        data={"error": f"连带崩溃: 遭致前置环节护盾击穿 {failed_deps_id}"}
                    )
                    results_map[task.task_id] = result
                    completed_count += 1
                    running_count -= 1
                    
                    for dependent_id in dependents.get(task.task_id, []):
                        pending_deps[dependent_id].discard(task.task_id)
                        if not pending_deps[dependent_id] and dependent_id not in results_map:
                            ready_queue.put_nowait(task_map[dependent_id])

                    ready_queue.task_done()
                    if completed_count >= total_count:
                        all_completed_event.set()
                        break
                    continue
                
                # Jinja2-style 简易跨越深渊变量灌注
                deps_data = {
                    dep_id: results_map[dep_id].data
                    for dep_id in task.depends_on
                }
                
                import re
                def render_template(val, scope_data):
                    if isinstance(val, str):
                        def replacer(match):
                            expr = match.group(1).strip()
                            parts = expr.split('.')
                            curr = scope_data
                            for p in parts:
                                if isinstance(curr, dict) and p in curr:
                                    curr = curr[p]
                                else:
                                    return match.group(0) 
                            return str(curr) if curr is not None else ""
                        new_val = re.sub(r'\{\{(.*?)\}\}', replacer, val)
                        return new_val
                    elif isinstance(val, dict):
                        return {k: render_template(v, scope_data) for k, v in val.items()}
                    elif isinstance(val, list):
                        return [render_template(v, scope_data) for v in val]
                    return val

                rendered_context = render_template(task.context, deps_data)
                
                execution_task = task
                if deps_data:
                    execution_task = task.model_copy(
                        update={
                            "context": {
                                **rendered_context,
                                "dependencies_data": deps_data
                            }
                        }
                    )

                child = BaseAgent(
                    role=self.role,
                    skills=self.skills,
                    max_depth=self.max_depth,
                    current_depth=self.current_depth + 1,
                    llm=self._llm,
                    reviewer=self._reviewer,
                )
                
                try:
                    result = await child.solve(execution_task)
                except (MaxRetryError, Exception) as e:
                    logger.error("编队坠落！子进程 %s 已阵亡: %s", task.task_id, e)
                    result = TaskResult(
                        task_id=task.task_id,
                        status="FAILED",
                        data={"error": str(e)},
                    )

                results_map[task.task_id] = result
                completed_count += 1
                running_count -= 1
                
                for dependent_id in dependents.get(task.task_id, []):
                    pending_deps[dependent_id].discard(task.task_id)
                    if not pending_deps[dependent_id] and dependent_id not in results_map:
                        ready_queue.put_nowait(task_map[dependent_id])

                ready_queue.task_done()
                if completed_count >= total_count:
                    all_completed_event.set()
                    break

        from config import MAX_CONCURRENT_SUBTASKS
        workers = [asyncio.create_task(worker()) for _ in range(MAX_CONCURRENT_SUBTASKS)]
        
        await all_completed_event.wait()
        for w in workers:
            w.cancel()

        return [results_map[t.task_id] for t in sub_tasks if t.task_id in results_map]

    async def _reduce(
        self,
        original_task: TaskRequest,
        sub_results: list[TaskResult],
        reduction_logic: str,
    ) -> TaskResult:
        """终局收网：拼合战报"""
        results_summary = [
            {
                "task_id": r.task_id,
                "status": r.status,
                "data": r.data,
            }
            for r in sub_results
        ]

        messages = [
            {
                "role": "system",
                "content": (
                    "你是一枚精湛的数据缝合梭机。根据指定的归纳逻辑合并子序列矩阵，"
                    "输出符合顶层 expected_output_schema 规范的纯正无瑕 JSON 载荷。\n"
                    "【严格输出约束】仅输出纯 JSON 对象，不得包含任何额外文字、注释或 markdown 标记。"
                    "所有 required 字段必须完整填写，字段名和类型必须严格匹配 Schema 定义。"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"## 标靶\n{original_task.goal}\n\n"
                    f"## 终端铸造模板\n{json.dumps(original_task.expected_output_schema, ensure_ascii=False)}\n\n"
                    f"## 锻造指令指南\n{reduction_logic}\n\n"
                    f"## 前线捷报汇总\n{json.dumps(results_summary, ensure_ascii=False)}\n"
                ),
            },
        ]

        reduced_data = await self._llm.chat_json(messages)

        failed = [r for r in sub_results if r.status == "FAILED"]
        status = "FAILED" if len(failed) == len(sub_results) else "SUCCESS"

        return TaskResult(
            task_id=original_task.task_id,
            status=status,
            data=reduced_data,
            artifacts=[a for r in sub_results for a in r.artifacts],
        )

    # ── 分支 C: UNKNOWN (探索模式) ──────────────────────────

    async def _branch_unknown(self, task: TaskRequest) -> TaskResult:
        """暗夜摸索：打探周边情报补充"""
        self._log("Dropping into UNKNOWN fog layer, initiating proactive intel gathering...")

        explore_task = task.model_copy()
        for attempt in range(1, MAX_EXPLORE_RETRIES + 1):
            self._log("Intel scouting sortie %d/%d", attempt, MAX_EXPLORE_RETRIES)
            explore_result = await self._explore(explore_task)

            if explore_result.has_enough_context:
                self._log("Intel payload acquired, diving back into the loop.")
                enriched = task.model_copy(
                    update={
                        "context": {
                            **task.context,
                            **explore_result.new_context,
                            "_explore_summary": explore_result.summary,
                        }
                    }
                )
                return await self.solve(enriched)

            explore_task = explore_task.model_copy(
                update={
                    "context": {
                        **explore_task.context,
                        f"_explore_attempt_{attempt}": explore_result.summary,
                    }
                }
            )

        self._log("Intel dry. Retiring request & summoning human commander.")
        return await self._ask_human(task)

    async def _explore(self, task: TaskRequest) -> ExploreResult:
        search_skill = self.skills.get("tavily_search") if self.skills else None
        if search_skill:
            self._log("Intel: Radar pinged, launching global web recon...")
            try:
                search_result = await search_skill.execute({"query": task.goal, "search_depth": "basic"})
                if "error" not in search_result:
                    self._log("Intel: Radar hits confirmed. Linking telemetry...")
                    return ExploreResult(
                        has_enough_context=True,
                        new_context={
                            "_search_results": search_result,
                            "_search_query": task.goal,
                        },
                        summary=f"已通过 tavily_search 猎获源发自「{task.goal}」的世界线资讯。",
                    )
                else:
                    self._log("Intel: Signal jamming: %s", search_result.get("error"))
            except Exception as e:
                self._log("Intel: Web recon crashed: %s", str(e))

        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个破译密码盲盒的推子。情报极度残缺缺失。\n"
                    "尝试脑补拼图，如果能靠自己拼凑够了，设 has_enough_context=true。\n"
                    "否则乖乖设 false 并尖叫你需要什么前置要素填补空白。\n"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"## 标靶\n- {task.goal}\n"
                    f"## 已有拼图\n{json.dumps(task.context, ensure_ascii=False)}\n"
                ),
            },
        ]
        return await self._llm.chat(messages, response_model=ExploreResult)

    async def _ask_human(self, task: TaskRequest) -> TaskResult:
        self._log("Human override invoked on ghost task %s", task.task_id)
        return TaskResult(
            task_id=task.task_id,
            status="FAILED",
            data={
                "error": "全频段扫描未果。请求拥有更高碳基维度的您介入指点。",
                "goal": task.goal,
            },
        )
