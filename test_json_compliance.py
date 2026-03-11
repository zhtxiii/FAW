import asyncio
import json
import traceback
from llm_client import LLMClient
from pydantic import BaseModel, Field

class TestSchema(BaseModel):
    name: str = Field(description="姓名")
    age: int = Field(description="年龄")
    hobbies: list[str] = Field(description="爱好")

async def test_json_output():
    client = LLMClient()
    print(f"Testing Model: {client._model}")
    
    # Test 1: Using response_model (Strict Schema)
    try:
        print("\n--- Test 1: Strict Pydantic Schema ---")
        messages1 = [{"role": "user", "content": "生成一个包含姓名小明、年龄18、爱好打篮球和敲代码的用户信息。"}]
        response1 = await client.chat(messages=messages1, response_model=TestSchema)
        print("Success! Parsed object:", response1)
    except Exception as e:
        print("Test 1 Failed:", e)
        traceback.print_exc()
        
    # Test 2: Using chat_json (Freeform JSON parsing)
    try:
        print("\n--- Test 2: chat_json Freeform ---")
        messages2 = [{"role": "user", "content": "请输出一个JSON对象，包含三个随机生成的英文单词在'words'列表中。必须只输出JSON。不要有任何多余的话。"}]
        response2 = await client.chat_json(messages=messages2)
        print("Success! Parsed dict:", response2)
    except Exception as e:
        print("Test 2 Failed:", e)
        traceback.print_exc()
        
    # Test 3: Complex JSON with nested structures
    try:
        print("\n--- Test 3: Complex nested JSON freeform ---")
        messages3 = [{"role": "user", "content": "请输出一个JSON对象，代表一个学校。包含属性：名称(string)、成立时间(int)、系(数组，每个元素包含系名和至少两个专业的列表)。除了JSON不要输出其他文字。"}]
        response3 = await client.chat_json(messages=messages3)
        print("Success! Parsed JSON:")
        print(json.dumps(response3, ensure_ascii=False, indent=2))
    except Exception as e:
        print("Test 3 Failed:", e)
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_json_output())
