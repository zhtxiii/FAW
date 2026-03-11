import asyncio
import json
import traceback
from openai import AsyncOpenAI
import os

# Override configurations for faster failure and visibility
os.environ["FAW_DEBUG_LLM"] = "1"
from config import OPENAI_API_KEY, OPENAI_BASE_URL, LLM_MODEL
from pydantic import BaseModel, Field

class TestSchema(BaseModel):
    name: str = Field(description="姓名")
    age: int = Field(description="年龄")
    hobbies: list[str] = Field(description="爱好")

async def test_json_output():
    client = AsyncOpenAI(
        api_key=OPENAI_API_KEY,
        base_url=OPENAI_BASE_URL,
        timeout=15.0, # Fail fast
        max_retries=0
    )
    print(f"Testing Model: {LLM_MODEL} at {OPENAI_BASE_URL}")

    # Test 1: Using response_format (if supported) or just prompt
    messages1 = [
        {"role": "system", "content": "You are a helpful assistant. You must output only raw valid JSON without markdown wrapping or extra text."},
        {"role": "user", "content": "生成一个包含姓名小明、年龄18、爱好打篮球和敲代码的用户信息。"}
    ]
    try:
        print("\n--- Test 1: Strict Prompt Schema ---")
        response = await client.chat.completions.create(
            model=LLM_MODEL,
            messages=messages1,
            max_tokens=2048
        )
        content = response.choices[0].message.content
        print("Raw Content:\n", content)
        
        # parse directly without pydantic first to see format compliance
        cleaned = content.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        
        parsed = json.loads(cleaned)
        print("Success! Parsed JSON:", parsed)
        
    except Exception as e:
        print("Test 1 Failed:", e)
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_json_output())
