import asyncio
from typing import Any
from skill_manager import Skill


class ExecuteCommandSkill(Skill):
    """在沙盒底层异步执行长耗时 Bash / CMD 命令并摘取流的指令抓手。"""

    name = "execute_command"
    description = "执行本地 Shell 命令（带15秒超时硬切防瘫痪）。输入 {'command': 'ls -al'}。切勿执行高危命令或引发阻断的互动程序。"

    async def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        cmd = params.get("command", "")
        if not cmd:
            return {"error": "没有收到合法的 'command' 串段。"}
            
        try:
            # 带超时的 async 子进程执行
            process = await asyncio.create_subprocess_shell(
                cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=15.0)
                out_str = stdout.decode('utf-8', errors='replace').strip()
                err_str = stderr.decode('utf-8', errors='replace').strip()
                
                result_map = {
                    "exit_code": process.returncode
                }
                if out_str:
                    result_map["stdout"] = out_str[:5000] + ("..." if len(out_str) > 5000 else "")
                if err_str:
                    result_map["stderr"] = err_str[:1000] + ("..." if len(err_str) > 1000 else "")
                
                return {"result": result_map}
            except asyncio.TimeoutError:
                process.kill()
                return {"error": f"操作耗时逾越 15 秒阈值被强制封杀了, 命令跑飞请检视: {cmd}"}
                
        except Exception as e:
            return {"error": f"唤起子进程壳槽抛锚: {str(e)}"}
