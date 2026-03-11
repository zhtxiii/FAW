import asyncio
import json
import logging
from base_agent import BaseAgent
from models import TaskRequest
from reviewer import CompositeReviewer, SchemaReviewer, HardcodeRuleReviewer, LLMReviewer
from skill_manager import create_default_registry

logging.basicConfig(level=logging.INFO)

async def main():
    with open("测试任务.txt", "r", encoding="utf-8") as f:
        goal = f.read().strip()

    task = TaskRequest(goal=goal, title="ROOT")
    registry = create_default_registry()
    reviewer = CompositeReviewer(
        schema_reviewer=SchemaReviewer(),
        hardcode_reviewer=HardcodeRuleReviewer(),
        semantic_reviewer=LLMReviewer()
    )
    agent = BaseAgent(max_depth=5, skills=registry, reviewer=reviewer)
    result = await agent.solve(task)
    print("================ FINAL RESULT ================")
    print(json.dumps(result.model_dump(), indent=2, ensure_ascii=False))

if __name__ == "__main__":
    asyncio.run(main())
