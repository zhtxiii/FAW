# FAW 深入运行示例与工作流解析

本文档通过具体案例，详细剖析 FAW（Fractal Agent Workflow）是如何处理自然语言指令的。我们将分别演示“简单任务（SIMPLE）”和“复杂任务（COMPLEX）”的处理过程，并深入解释**技能调用（Skills）**与**三重审核（Schema Review）**机制及核心的数据交互格式。

---

## 示例 1：简单任务 (SIMPLE) 与动态技能调用

### 场景描述
**用户指令**：“帮我获取并计算苹果公司（AAPL）目前的股票市盈率（PE Ratio），假设当前股价是 170，每股收益是 5.2。”

### 1. 输入解析与单次规划 (One-Shot Planning)
核心大脑（`base_agent.py`）接收到指令后，单次 LLM 推理评估发现该任务只需一步计算即可完成，无需拆解。

**LLM 生成并返回的 `ExecutionPlannerResult` 契约数据：**
```json
{
  "decision": "SIMPLE",
  "simple_plan": {
    "use_skill": true,
    "skill_name": "math_calculator",
    "skill_params": {
      "expression": "170 / 5.2"
    },
    "direct_result": null
  },
  "sub_tasks": null,
  "reduction_logic": "",
  "new_context_request": ""
}
```

> **🔑 核心字段解析与 FAW 特性说明：**
> *   `decision`: 路由决策。`SIMPLE` 表示这是单一节点任务，无需 Map-Reduce 拆解。
> *   `simple_plan`: 简单任务执行计划，指明如何完成这一步。
> *   `use_skill`: 是否需要调用外部工具/技能。
> *   `skill_name` 与 `skill_params`: 指定调用的内置工具名称与传参。此时 LLM 选择调用工具库中已有的 `math_calculator` 执行具体的除法运算。

> **💡 进阶：如果工具库里没有对应的工具怎么办？**
> 规划器 LLM **不需要编写任何代码**，它只需声明一个语义化的工具名称（例如 `"stock_pe_calculator"`）和参数即可。
> 当 Skill Manager 发现该工具不存在时，系统会自动触发一个**专职的开发智能体（Skill Synthesizer）**：
> 1. 开发智能体接收到工具名称、参数样例和任务背景信息。
> 2. 开发智能体使用专精的代码编写 Prompt 生成完整的 Python 工具类代码。
> 3. 系统在沙箱中编译验证该代码，确保其合法性。
> 4. 验证通过后，技能被注册到内存中并**持久化**写入 `skills/` 目录，成为永久资产。
>
> 这种"**职责分离**"的设计确保了规划器 LLM 只需专注于任务拆解和工具选择，而代码编写的工作由专职智能体承担，大幅提升了系统推理的稳定性和生成代码的质量。

### 2. 技能执行 (Skill execution)
系统查阅 `simple_plan`，发现任务需要调用预置的 `math_calculator` 技能。
**Skill Manager（技能加载与调度中心）** 会提取 `expression` 参数并传入底层工具。计算完成后，工具返回提取到的结果 `32.69`。

### 3. 构建终态返回
节点执行成功后，将计算结果按标准封装，构建为一致的终态契约 `TaskResult`:
```json
{
  "task_id": "a1b2c3d4e5f6",
  "status": "SUCCESS",
  "data": {
    "pe_ratio": 32.69,
    "message": "AAPL PE Ratio calculated successfully."
  },
  "artifacts": [] // 若任务生成了本地图表或文件，所保存的绝对路径会存放在这里
}
```

---

## 示例 2：复杂任务 (COMPLEX) 的多层拆解与聚合

### 场景描述
**用户指令**：“帮我调研一下目前市场上两款 AI 编程助手（Cursor 和 GitHub Copilot）的定价策略，分别统计他们的优缺点，最后给我输出一份 Markdown 格式的对比报告。”

### 1. 单次规划与裂变 (Map)
核心大脑评估认为该任务涉及多次数据获取与整合，判定为 `COMPLEX` 难度，需要拆解。

**LLM 生成的 `ExecutionPlannerResult` 数据：**
```json
{
  "decision": "COMPLEX",
  "simple_plan": null,
  "reduction_logic": "将 Cursor 和 Copilot 的调研数据提取出来，梳理对比其定价和优缺点，最后格式化为 Markdown 文档返回。",
  "sub_tasks": [
    {
      "task_id": "task_cursor_research",
      "title": "调研 Cursor 的定价与优缺点",
      "goal": "搜索并总结 Cursor 编程助手的定价方案及核心优缺点",
      "expected_output_schema": {
        "pricing": "string",
        "pros": ["string"],
        "cons": ["string"]
      },
      "depends_on": [],
      "validation_requirements": ["返回结果必须包含具体的美元价格数字"],
    },
    {
      "task_id": "task_copilot_research",
      "title": "调研 GitHub Copilot 的定价与优缺点",
      "goal": "搜索并总结 GitHub Copilot 的定价方案及核心优缺点",
      "expected_output_schema": {
        "pricing": "string",
        "pros": ["string"],
        "cons": ["string"]
      },
      "depends_on": [],
      "validation_requirements": ["返回结果必须包含具体的美元价格数字"],
    }
  ]
}
```

> **🔑 复杂任务核心字段解析：**
> *   `decision`: `COMPLEX` 标识当前任务无法一步完成，需要开启递归拆解树。
> *   `sub_tasks`: 包含裂变出的子任务数组，每个子任务自身都是一个完整的 `TaskRequest` 请求体。
> *   `depends_on`: DAG (有向无环图) 拓扑依赖控制。如果任务 B 依赖 A，此处会填写 A 的 `task_id`，系统调度器据此进行排队阻塞。本例中由于均为空 `[]`，系统将直接采取**并发（异步）**方式同时全速运行这两个爬虫/调研节点。
> *   `expected_output_schema`: **最关键的防御契约**。系统强制要求该节点必须产出完全符合该字典结构的 JSON。这也是应对大模型格式幻觉，触发后续“统一验收”的第一道强制防线。
> *   `reduction_logic`: Reduce 环节的“合并说明书”。当底下所有子任务都成功完成并返回后，主节点将根据这段提示语，把散落的多个 `TaskResult.data` 缝合为一份最终交付产物。

### 2. 并发调度 (Concurrent Execution)
因为两个子任务的 `depends_on` 均为空列表 `[]`，FAW 会**并发（异步）**启动两个子节点，它们可能各自调用内置的 `tavily_search` 技能去联网获取资料。

### 3. 统一验收机制 (Unified Validation)
假设 `task_cursor_research` 执行完毕节点大模型返回了如下原始结果：
```json
{
  "pricing": "Pro版 $20/月",
  "pros": ["集成Claude 3.5", "代码自动补全体验极佳"],
  "cons": "有时候服务器会卡顿"
}
```

此时会触发 FAW 的统一验收机制：
1. **Schema 格式校验**：
   系统发现预期的 `expected_output_schema` 中，`cons` 应该是一个字符串列表 `["string"]`，但模型返回了单行字符串。格式校验不通过，触发重试或自动修复。
2. **代码验证（代码生成模型自动翻译）**：
   统一验收器调用专职代码生成模型，将自然语言需求 `"返回结果必须包含具体的美元价格数字"` 翻译为 Python 代码：`"'$' in data.get('pricing', '') and any(c.isdigit() for c in data.get('pricing', ''))"`。代码执行结果确认 `"Pro版 $20/月"` 中含有美元符号和数字，验证通过。
3. **LLM 兆底（仅当代码无法覆盖时）**：
   如果存在无法用代码表达的主观性验证需求（如"分析要有深度"），则兑底调用 LLM 进行语义审核。

只有闯过所有验收关卡，节点才被标记为 `SUCCESS`，并输出最终的 `TaskResult`。

### 4. 缝合与归结 (Reduction)
两个子任务纷纷并成功输出合规的 `TaskResult` 后，系统引擎触发 **Reduce** 操作：
将两份 `TaskResult.data` 和一阶段设定的 `reduction_logic` 喂入汇总代理中。

汇总代理理解指令并输出终极的 `TaskResult`：
```json
{
  "task_id": "main_compare_task",
  "status": "SUCCESS",
  "data": {
    "report_content": "# AI 编程助手对比分析\n\n## 1. 定价策略\n- **Cursor**: Pro版 $20/月...\n- **GitHub Copilot**: 个人版 $10/月...\n\n## 2. 优缺点\n..."
  },
  "artifacts": []
}
```
此时，整个树状推演彻底完结。流式前端界面上的两根树杈将融合变绿，任务成功终止。

---

## 示例 3：探索任务 (UNKNOWN) 的试探与信息收集

### 场景描述
**用户指令**：“目前世界上有没有可以直接用脑电波操控编写 Python 全栈应用且已正式商用的脑机接口设备？如果没有，现在最前沿的做到哪一步了，请给我列出这些型号。”

### 1. 意图不明与打探 (Unknown)
主控大脑在接收到这个极度前沿、偏冷门且信息量不明确的任务后，发现仅仅凭借现在的 Context 以及内部系统预置规则根本不足以直接把任务裂变为诸如“查询发售价格”、“总结功能”等结构化分支。贸然发起 `COMPLEX` 极易导致疯狂幻觉。
因此，规划器主动退防，判定为 `UNKNOWN` 难度。

**LLM 生成的 `ExecutionPlannerResult` 数据：**
```json
{
  "decision": "UNKNOWN",
  "simple_plan": null,
  "sub_tasks": null,
  "reduction_logic": "",
  "new_context_request": "系统当前上下文中缺乏关于“商业化脑机接口编程设备”的任何最新情报。请立即调用全局搜索引擎粗略调查 'commercial brain-computer interface for coding Python full-stack' 的技术现状，了解是否存在真实商用的方案或处于何种实验室阶段，然后再把情报交由我重新制订完整计划。"
}
```

> **🔑 探索分支核心字段解析：**
> *   `decision`: `UNKNOWN` 标识系统遭遇了知识盲区，或认为指令包含的矛盾假设极多，贸然拆解必然导致严重的逻辑雪崩。
> *   `new_context_request`: 系统不会直接摆烂中止，而是主动要求上级调度器提供额外的前置“知识补给包”。
>   
> FAW 底座在接收到这条 `UNKNOWN` 撤退请求后，会自动把这行字转化为一个临时的 `SIMPLE` 数据挂载任务（比如静默调用 `tavily_search` 去摸排大局），待侦察兵拿回基础知识（Context）后，再连同收集到的新情报**重新启动一轮 One-Shot Planning**。此时，有了底气的规划器就会顺利切入 `COMPLEX` 分支，精准指派后续兵团。
