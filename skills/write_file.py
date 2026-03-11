import os
from typing import Any
from skill_manager import Skill


class WriteFileSkill(Skill):
    """执行文件写出的核心生产力。"""

    name = "write_file"
    description = "将输出文本覆写至指明文件中。输入 {'file_path': '待生成目录/档案.txt', 'content': '想要存放的内容'}。如果目录缺失会自动补全建立。"

    async def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        file_path = params.get("file_path", "")
        content = params.get("content", "")
        
        if not file_path:
            return {"error": "需要写入位置参量 'file_path'。"}
            
        try:
            dir_name = os.path.dirname(file_path)
            if dir_name and not os.path.exists(dir_name):
                os.makedirs(dir_name, exist_ok=True)
                
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(str(content))
            return {"result": f"数据已被持久化装载于底盘: {file_path}"}
        except Exception as e:
            return {"error": f"写入落盘时触发异常: {str(e)}"}
