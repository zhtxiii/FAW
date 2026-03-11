"""FAW LLM 通信层。

封装与 OpenAI 兼容 API 的所有交互，提供类型安全的结构化输出。
"""

from __future__ import annotations

import json
import logging
import httpx
from typing import Any, Type, TypeVar

from openai import AsyncOpenAI
from pydantic import BaseModel

from config import OPENAI_API_KEY, OPENAI_BASE_URL, LLM_MODEL

logger = logging.getLogger(__name__)
# 专属业务流 Logger
io_logger = logging.getLogger("[LLM_IO]")

T = TypeVar("T", bound=BaseModel)


class LLMClient:
    """异步 LLM 客户端，封装 OpenAI 兼容 API 调用。"""

    def __init__(self, model: str | None = None):
        self._model = model or LLM_MODEL
        self._client = AsyncOpenAI(
            api_key=OPENAI_API_KEY,
            base_url=OPENAI_BASE_URL,
            timeout=120.0,
            max_retries=3,
        )

    async def chat(
        self,
        messages: list[dict],
        response_model: Any | None = None,
        temperature: float = 0.3,
    ) -> Any:
        """通用对话方法。

        Args:
            messages: OpenAI 格式的消息列表。
            response_model: 可选的 Pydantic 模型，用于将 JSON 响应解析为结构化对象。
            temperature: 生成温度。

        Returns:
            原始字符串或解析后的 Pydantic 模型实例。
        """
        import config
        if config.DEBUG_LLM:
            msg = f"🤖 [LLM_REQ] Model: {self._model}\n"
            for m in messages:
                content_snip = m['content'][:300] + '...' if len(m['content']) > 300 else m['content']
                msg += f"   [{m['role'].upper()}]: {content_snip}\n"
            io_logger.debug(msg.strip())

        if response_model is not None:
            # 引导 LLM 输出符合 schema 的 JSON
            schema_hint = json.dumps(
                response_model.model_json_schema(), ensure_ascii=False, indent=2
            )
            system_suffix = (
                f"\n\n【输出格式强制约束】你必须严格按以下 JSON Schema 输出，不要包含 markdown 代码块标记，仅输出纯 JSON。"
                f"不得输出任何额外文本、解释或注释，仅输出合法 JSON 对象。"
                f"所有字段必须按 Schema 要求填写，缺失字段会导致系统崩溃：\n{schema_hint}"
            )
            # 将 schema 提示追加到 system 消息中
            if messages and messages[0]["role"] == "system":
                messages = [
                    {**messages[0], "content": messages[0]["content"] + system_suffix},
                    *messages[1:],
                ]
            else:
                messages = [
                    {"role": "system", "content": f"你是一个智能助手。{system_suffix}"},
                    *messages,
                ]

        logger.debug("LLM request: model=%s, messages=%d", self._model, len(messages))

        # deepseek-reasoner 不支持 temperature 参数
        create_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": 8192,
        }
        if "reasoner" not in self._model:
            create_kwargs["temperature"] = temperature

        response = await self._client.chat.completions.create(**create_kwargs)  # type: ignore[arg-type]
        raw_content = response.choices[0].message.content or ""
        finish_reason = response.choices[0].finish_reason
        if finish_reason == "length":
            logger.warning("LLM response truncated due to hitting max_tokens limit! Output may be incomplete JSON.")
        
        import config
        if config.DEBUG_LLM:
            io_logger.debug(f"🟢 [LLM_RES]: {raw_content}")

        if response_model is not None:
            cleaned = raw_content.strip()
            if cleaned.startswith("```"):
                lines = cleaned.split("\n")
                cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
            return response_model.model_validate_json(cleaned)

        return raw_content

    async def chat_json(
        self,
        messages: list[dict],
        temperature: float = 0.3,
    ) -> dict[str, Any]:
        """对话并返回解析后的 JSON dict。"""
        import config
        if config.DEBUG_LLM:
            msg = f"🤖 [LLM_JSON_REQ] Model: {self._model}\n"
            for m in messages:
                content_snip = m['content'][:300] + '...' if len(m['content']) > 300 else m['content']
                msg += f"   [{m['role'].upper()}]: {content_snip}\n"
            io_logger.debug(msg.strip())

        raw = await self.chat(messages, temperature=temperature)
        assert isinstance(raw, str)
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            cleaned = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        return json.loads(cleaned)
