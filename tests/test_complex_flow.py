import asyncio
import json
from base_agent import BaseAgent
from llm_client import LLMClient
from reviewer import CompositeReviewer, SchemaReviewer, HardcodeRuleReviewer, LLMReviewer
from skill_manager import create_default_registry
from models import TaskRequest

async def main():
    llm = LLMClient()
    skills = create_default_registry()
    reviewer = CompositeReviewer(
        SchemaReviewer(),
        HardcodeRuleReviewer(),
        LLMReviewer(llm)
    )
    agent = BaseAgent(llm=llm, skills=skills, reviewer=reviewer)
    task = TaskRequest(
        goal="搜索一下今天关于AI的前两篇新闻，然后针对影响力最大的一条展开分析，最后给出一篇简短两句话总结",
    )
    res = await agent.solve(task)
    print("FINAL STATUS:", res.status)
    if res.status == "SUCCESS":
        print(json.dumps(res.data, indent=2, ensure_ascii=False))
    else:
        print("ERROR:", res.data)

if __name__ == "__main__":
    asyncio.run(main())
