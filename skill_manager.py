"""FAW 技能加载与管理模块。

提供高度动态化、按需插拔的 Skills 加载体系。取代原有的静态 tools 系统，
每个 Skill 将是一个独立的纯模块实体，系统仅在需要时将其加载和挂载到内存。
"""

from __future__ import annotations

import importlib
import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)
skills_logger = logging.getLogger("[SKILLS]")


class Skill(ABC):
    """技能基类。所有新技能必须继承此类。"""

    name: str = ""
    description: str = ""

    @abstractmethod
    async def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        """执行环境交互动作或运算。

        Args:
            params: 技能特供的参数字典。

        Returns:
            执行结果字典。
        """
        ...


class SkillManager:
    """技能管理器，负责维护当前已启用的可用技能映射，并按名字解析调用。"""

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        """注册一个实例化的技能对象。"""
        self._skills[skill.name] = skill
        from config import DEBUG_SKILLS
        if DEBUG_SKILLS:
            skills_logger.debug(f"🔌 [技能挂载] 成功装载技能组件: {skill.name}")

    def load_from_code(self, skill_code: str, class_name: str) -> Skill:
        """从原生注入的代码中动态编译及装载（如 LLM 自制的临时技能）。"""
        from config import DEBUG_SKILLS
        if DEBUG_SKILLS:
            skills_logger.debug(f"⚙️ [动态生成] 尝试编译并构建临时技能类: {class_name}")

        namespace: dict[str, Any] = {
            "Skill": Skill,
            "Any": Any,
        }
        try:
            exec(skill_code, namespace)
            skill_class = namespace.get(class_name)
            if not skill_class or not issubclass(skill_class, Skill):
                raise ValueError(f"代码中未能正确解析出受派生保护的技能子类: {class_name}")

            skill_instance = skill_class()
            self.register(skill_instance)
            return skill_instance
        except Exception as e:
            logger.error("加载动态生成的技能破损: %s", e)
            raise ValueError(f"技能代码非法或抛错: {str(e)}")

    def get(self, name: str) -> Skill | None:
        """检索挂载点中指定的技能对象。"""
        return self._skills.get(name)

    def list_skills(self) -> list[dict[str, str]]:
        """回吐当前可用装载所有技能的能力清单，以供路由或下发评估上下文使用。"""
        return [
            {"name": s.name, "description": s.description}
            for s in self._skills.values()
        ]

    @property
    def skill_names(self) -> list[str]:
        return list(self._skills.keys())

    async def synthesize_skill(
        self,
        skill_name: str,
        skill_params: dict,
        task_goal: str,
        expected_output_schema: dict | None = None,
        llm: Any = None,
    ) -> Skill | None:
        """当请求的技能不存在时，调用专职开发 LLM 合成该技能代码。

        流程：
          1. 向开发智能体发送专精的代码编写 Prompt
          2. 接收交付的 Python 代码
          3. 编译并验证代码合法性
          4. 注册到内存并持久化到 skills/ 目录

        Args:
            skill_name: 规划器请求的技能名称。
            skill_params: 规划器传入的参数样例，帮助开发智能体理解接口。
            task_goal: 原始任务目标，为代码编写提供语境。
            llm: LLM 客户端实例。

        Returns:
            合成成功时返回已注册的 Skill 实例，失败返回 None。
        """
        from models import SkillSynthesisResult

        if llm is None:
            from llm_client import LLMClient
            llm = LLMClient()

        skills_logger.info(
            "🔧 [技能合成] 工具库中不存在 '%s'，启动开发智能体进行合成...", skill_name
        )

        # 构建专精的代码编写 Prompt
        import json
        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个专职的 Python 开发工程师。你的唯一任务是编写一个继承自 Skill 基类的工具类。\n"
                    "严格要求：\n"
                    "1. 类必须继承 Skill 基类（已在环境中可用），并实现 async def execute(self, params: dict[str, Any]) -> dict[str, Any] 方法。\n"
                    "2. 类必须设置 name 和 description 属性。\n"
                    "3. 代码必须包含完整的异常处理（try-except），失败时返回包含 'error' 键的字典。\n"
                    "4. 不要使用任何需要 pip 安装的第三方库。仅使用 Python 标准库。\n"
                    "5. 仅输出纯 Python 代码，不要包含 markdown 代码块标记或任何解释文字。\n"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"请编写一个名为 '{skill_name}' 的技能工具类。\n"
                    f"任务背景: {task_goal}\n"
                    f"期望的参数接口示例: {json.dumps(skill_params, ensure_ascii=False)}\n"
                    + (f"期望的输出Schema: {json.dumps(expected_output_schema, ensure_ascii=False)}\n请确保返回的字典结构严格符合该 Schema 要求。\n\n" if expected_output_schema else "\n")
                    + "请直接输出完整的 Python 类定义代码。"
                ),
            },
        ]

        try:
            raw_code = await llm.chat(messages, temperature=0.2)
            assert isinstance(raw_code, str), "开发智能体未返回有效代码字符串"

            # 清理可能的 markdown 代码块包裹
            cleaned = raw_code.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                cleaned = "\n".join(
                    lines[1:-1] if lines[-1].strip() == "```" else lines[1:]
                )

            # 从代码中提取类名（找到继承 Skill 的类）
            import re
            class_match = re.search(r"class\s+(\w+)\s*\(.*Skill.*\)", cleaned)
            if not class_match:
                logger.error("[技能合成] 生成的代码中无法找到继承 Skill 的类定义")
                return None
            class_name = class_match.group(1)

            # 编译并注册
            skill_instance = self.load_from_code(cleaned, class_name)

            # 持久化到磁盘
            self._persist_skill(skill_name, cleaned)

            skills_logger.info(
                "✅ [技能合成] 成功合成并注册技能 '%s' (类: %s)", skill_name, class_name
            )
            return skill_instance

        except Exception as e:
            logger.error("[技能合成] 合成技能 '%s' 失败: %s", skill_name, e)
            return None

    @staticmethod
    def _persist_skill(skill_name: str, code: str) -> None:
        """将合成的技能代码持久化写入 skills/ 目录。"""
        import os
        skills_dir = os.path.join(os.path.dirname(__file__), "skills")
        os.makedirs(skills_dir, exist_ok=True)
        filepath = os.path.join(skills_dir, f"{skill_name}.py")

        # 如果文件已存在则不覆盖（避免意外覆写内置技能）
        if os.path.exists(filepath):
            logger.warning("[技能持久化] 文件 %s 已存在，跳过写入", filepath)
            return

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(code)
        skills_logger.info("💾 [技能持久化] 已保存至 %s", filepath)


def create_default_registry() -> SkillManager:
    """初始化装载了环境四大核心能力与俩计算文本扩展能力的默认池。"""
    manager = SkillManager()
    
    # 动态按需加载模块实体 (避免启动时无意义地吃内存)
    def _bootstrap_module(module_name: str, class_name: str):
        try:
            mod = importlib.import_module(f"skills.{module_name}")
            cls = getattr(mod, class_name)
            manager.register(cls())
        except Exception as e:
            logger.error(f"严重错误：组件技能 '{module_name}' 加载损坏 -> {e}")

    # 基础内置扩展模块
    _bootstrap_module("calculator", "CalculatorSkill")
    _bootstrap_module("text_processor", "TextProcessorSkill")
    
    # 获取环境互操作和云联接的四大核心模块
    _bootstrap_module("read_file", "ReadFileSkill")
    _bootstrap_module("write_file", "WriteFileSkill")
    _bootstrap_module("execute_command", "ExecuteCommandSkill")
    _bootstrap_module("tavily_search", "TavilySearchSkill")
    
    return manager
