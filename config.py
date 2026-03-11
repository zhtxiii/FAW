"""FAW 全局配置模块。

集中管理所有配置常量，从环境变量读取 LLM 相关设置。
"""

import os

# ── 重试与深度限制 ──────────────────────────────────────────
MAX_RETRIES: int = 3            # 审核不通过的最大重试次数
MAX_EXPLORE_RETRIES: int = 3    # 探索模式最大重试次数
DEFAULT_MAX_DEPTH: int = 5      # 默认最大递归深度

# ── LLM 配置 ───────────────────────────────────────────────
LLM_MODEL: str = os.getenv("FAW_LLM_MODEL", "deepseek-v3.2")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL: str | None = os.getenv("OPENAI_BASE_URL", "https://poloapi.top/v1")

# ── 并发配置 ───────────────────────────────────────────────
MAX_CONCURRENT_SUBTASKS: int = int(os.getenv("FAW_MAX_CONCURRENT", "10"))

# ── 调试细分配置 ───────────────────────────────────────────
DEBUG_LLM: bool = os.getenv("FAW_DEBUG_LLM", "0") == "1"
DEBUG_TASKS: bool = os.getenv("FAW_DEBUG_TASKS", "1") == "1"
DEBUG_SKILLS: bool = os.getenv("FAW_DEBUG_SKILLS", "1") == "1"
