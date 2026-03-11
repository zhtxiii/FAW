import os
from typing import Any
from skill_manager import Skill


class ReadFileSkill(Skill):
    """读取本地文件内容的系统级能力。"""

    name = "read_file"
    description = "读取本地路径下的文件数据摘要。输入 {'file_path': '/path/to/file'} 返回其文本。请小心使用绝对路径或基于工作区的相对路径。"

    async def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        file_path = params.get("file_path", "")
        if not file_path:
            return {"error": "缺少 'file_path' 参数。"}
        
        if not os.path.exists(file_path):
            return {"error": f"找不到指名文件: {file_path}"}
            
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            # 过于庞大的数据要截断，以免压塌 LLM Token 限制
            if len(content) > 10000:
                content = content[:10000] + "...(文件体积过大已截断显示)"
            return {"result": content}
        except Exception as e:
            return {"error": f"无法读取给定源文件: {str(e)}"}
