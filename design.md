架构蓝图：递归 Map-Reduce 智能体网络 (Fractal Agent Workflow)
1. 架构概述 (Architecture Overview)
请实现一套支持无限递归嵌套的多智能体系统。系统的核心逻辑是：任何一个智能体在接收到任务时，都可以动态决定是“自行处理”、“Map-Reduce 拆解下发”还是“进入探索模式”。
为了保证输出的绝对稳定性，系统中所有智能体的输出，都必须经过一个前置的“审核智能体 (Reviewer Agent) / 验证函数”拦截，审核通过后才允许向上层交付。

2. 核心实体设计 (Core Entities)
请在代码中实现以下三个核心基类/模块：

2.1 BaseAgent (执行节点基类)
所有智能体（不管是顶层还是第N层）都继承自此基类。

属性: agent_id, role, tools (挂载的工具列表), max_depth (最大递归深度), current_depth (当前深度)。

核心方法: solve(task_request) -> 返回 TaskResult。

2.2 PlannerRouter (决策路由引擎)
内置于每个 BaseAgent 中。收到任务后，调用大模型进行评估，输出状态机的下一步走向（枚举值）：

SIMPLE: 任务简单，当前工具足够，直接执行。

COMPLEX: 任务复杂，需要拆解（Map-Reduce）。

UNKNOWN: 缺乏前置信息，不知如何下手，进入探索状态。

2.3 Reviewer / Validator (审核守门员)
独立于执行逻辑。可以是一个简单的 Python Assert 函数，也可以是一个专门的轻量级 LLM 智能体。

输入: original_task, generated_result

输出: {"passed": boolean, "feedback": string}

3. 核心状态机与工作流 (The Core State Machine)
请在 BaseAgent.solve() 方法中实现以下控制流：

步骤 1: 路由评估 (Routing)
使用 PlannerRouter 评估 task_request。

步骤 2: 分支执行 (Branching)

分支 A (状态: SIMPLE):

智能体直接调用自身挂载的工具完成任务。

产生临时结果 temp_result。

分支 B (状态: COMPLEX - Map-Reduce 逻辑):

防死锁检查: 如果 current_depth >= max_depth，强制降级为分支 A（尽力而为）或抛出异常。

Map: 调用大模型，将任务拆解为包含多个子任务的 JSON 数组。

Dispatch: 针对每个子任务，动态克隆或实例化新的子 BaseAgent(current_depth + 1)，并行调用子节点的 solve() 方法。

Reduce: 收集所有子节点的返回结果，交由当前智能体进行汇总总结，产生 temp_result。

分支 C (状态: UNKNOWN - 探索逻辑):
*

进入 explore_mode。智能体执行试探性搜索、查阅基础文档，或将子问题下发给子智能体执行。

如果探索后获得了足够上下文：将新上下文附加到原任务中，递归调用自身的 solve() (转回步骤 1)。

如果探索重试 3 次仍无进展：触发 ask_human() 接口（如果在无人值守模式下，则向上抛出 Failed）。

步骤 3: 强制审核与重试 (Review & Retry Loop)

将 temp_result 提交给 Reviewer。

如果 Reviewer.passed == True: 结束执行，正式返回结果。

如果 Reviewer.passed == False:

触发重试逻辑。将 Reviewer.feedback 作为新的输入上下文，要求当前逻辑（分支 A 或 B）重新生成。

如果 retry_count > MAX_RETRIES: 抛出 MaxRetryError，向父节点宣告本节点任务失败。

4. 数据契约设计 (Data Contracts / JSON Schemas)
为了防止递归过程中的上下文崩溃，请使用 Pydantic (或类型提示) 严格定义以下数据结构。智能体之间的通信仅限使用这些结构：

Python
# 1. 任务输入契约
class TaskRequest(BaseModel):
    task_id: str
    goal: str
    context: dict  # 仅包含该节点所需的最小上下文，严禁将父节点的无用历史传入
    expected_output_schema: dict # 极其重要：要求子节点交付的具体数据格式

# 2. Map 拆解契约 (主管智能体拆分任务时的输出)
class SubTaskPlan(BaseModel):
    sub_tasks: List[TaskRequest]
    reduction_logic: str # 指导 Reduce 阶段如何把结果拼起来的说明

# 3. 最终交付契约
class TaskResult(BaseModel):
    task_id: str
    status: Literal["SUCCESS", "FAILED"]
    data: dict # 必须严格符合 TaskRequest 中的 expected_output_schema
    artifacts: List[str] # 产生的临时文件或代码路径
5. 关键工程约束 (Engineering Constraints)
自动化编程工具在编写代码时，必须遵守：

绝对的上下文隔离： 子节点在初始化时，context 必须是干净的，只能包含父节点在 Map 阶段明确指定的信息。

异步并发： 分支 B 的 Dispatch 阶段，必须使用 asyncio.gather 或线程池来并发执行无依赖关系的子任务。