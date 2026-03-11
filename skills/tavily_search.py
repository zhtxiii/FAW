import io
import json
import urllib.request
import urllib.parse
from typing import Any
from skill_manager import Skill


class TavilySearchSkill(Skill):
    """赋予大模型洞察全网情报底蕴的搜索引擎对接基石模块。"""

    name = "tavily_search"
    description = ("使用 Tavily 进行互联网搜索补充知识空白。输入 {'query': 'XXX', 'search_depth': 'basic'} 获得聚合的问询解答。"
                   "强烈建议在 'UNKNOWN' 遭遇知识瓶颈时果断调用本接口进行全网探寻。"
                   "返回格式: {'answer': '聚合回答文本', 'hits': [{'title': '标题', 'url': '链接', 'content': '内容摘要'}, ...]}")
    
    _API_KEY = "REDACTED_TAVILY_API_KEY"
    _ENDPOINT = "https://api.tavily.com/search"

    async def execute(self, params: dict[str, Any]) -> dict[str, Any]:
        query = params.get("query", "")
        if not query:
            return {"error": "缺少 'query' 关键搜索词意图。"}
            
        search_depth = params.get("search_depth", "basic")
        
        req_data = {
            "api_key": self._API_KEY,
            "query": query,
            "search_depth": search_depth,
            "include_answer": True,
            "include_raw_content": False,
            "max_results": 5,
        }
        
        req_body = json.dumps(req_data).encode("utf-8")
        request = urllib.request.Request(
            self._ENDPOINT,
            data=req_body,
            headers={
                "Content-Type": "application/json"
            },
            method="POST"
        )
        
        try:
            # 由于大框架目前主要是 asyncio 并发架构，而 urllib 是阻塞的，但作为轻度调用能将就
            # 真实场景应该用 aiohttp，但这能规避多装一个依赖。
            import asyncio
            loop = asyncio.get_running_loop()
            
            def _do_req():
                with urllib.request.urlopen(request, timeout=10.0) as resp:
                    resp_body = resp.read()
                    return json.loads(resp_body)
                    
            resp_json = await loop.run_in_executor(None, _do_req)
            
            # 返回扁平结构，便于规划器 schema 适配
            return {
                "answer": resp_json.get("answer", ""),
                "hits": [
                    {"title": r.get("title", ""), "url": r.get("url", ""), "content": r.get("content", "")} 
                    for r in resp_json.get("results", [])
                ]
            }
        except Exception as e:
            return {"error": f"搜索连接引擎报错失联: {str(e)}"}
