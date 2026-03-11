from typing import Any
from skill_manager import Skill


class TextProcessorSkill(Skill):
    """文本处理与清洗技能，提供基本的词量解析和反转功能。"""

    name = "text_processor"
    description = (
        "基础文本处理技能。支持操作: count_words(统计词数), to_upper(转大写), to_lower(转小写), reverse(反转)。\n"
        "输入格式: {'operation': '...', 'text': '...'}\n"
        "🚨 注意：此工具的输出格式严格固定为 {'result': <string or int>}。它不支持任何复杂的结构化提取或字典输出。\n"
        "如果任务的 expected_output_schema 包含诸如 title, content, reason 等复杂字段，切勿使用此工具！"
    )

    async def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        op = params.get("operation", "")
        text = str(params.get("text", ""))

        if op == "count_words":
            # 兼容老版测试的中英文断词器（暂简易依据空格，或全长判断）
            words = text.split()
            return {"result": len(words) if words else len(text)} 
        elif op == "to_upper":
            return {"result": text.upper()}
        elif op == "to_lower":
            return {"result": text.lower()}
        elif op == "reverse":
            return {"result": text[::-1]}
        else:
            return {"error": f"未知操作: {op}"}
