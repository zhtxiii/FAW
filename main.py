"""FAW 入口与示例。

提供 CLI 入口，创建根 Agent 并执行任务。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys

from base_agent import BaseAgent, MaxRetryError
from config import DEFAULT_MAX_DEPTH
from models import TaskRequest
from reviewer import CompositeReviewer, HardcodeRuleReviewer, LLMReviewer, SchemaReviewer
from skill_manager import create_default_registry


def setup_logging(
    debug_llm: bool = False,
    debug_tasks: bool = False,
    debug_skills: bool = False,
    log_file: str | None = None
) -> None:
    """配置日志记录，根据不同组件的 debug 开关调整全局表现。"""
    import config
    config.DEBUG_LLM = debug_llm or config.DEBUG_LLM
    config.DEBUG_TASKS = debug_tasks or config.DEBUG_TASKS
    config.DEBUG_SKILLS = debug_skills or config.DEBUG_SKILLS

    base_level = logging.DEBUG if any([debug_llm, debug_tasks, debug_skills]) else logging.INFO
    
    # 我们定制一个格式，淡化常规的 DEBUG/INFO，强化业务模块名 (name) 作为主题标签！
    fmt = "%(asctime)s | %(name)-13s | %(message)s"
    formatter = logging.Formatter(fmt)
    
    handlers: list[logging.Handler] = []
    
    # 终端输出
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    handlers.append(stream_handler)
    
    # 文件输出
    if log_file:
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)

    # 先清除系统已有的 handlers 然后重新挂载
    root_logger = logging.getLogger()
    root_logger.setLevel(base_level)
    for h in root_logger.handlers[:]:
        root_logger.removeHandler(h)
    for h in handlers:
        root_logger.addHandler(h)
    
    # 降低底层第三方库造成的噪音
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    
    # 如果特定维度未开启，则压制相关模块的展示
    if not debug_llm:
        logging.getLogger("[LLM_IO]").setLevel(logging.INFO)
    if not debug_skills:
        logging.getLogger("[SKILLS]").setLevel(logging.INFO)
    if not debug_tasks:
        logging.getLogger("[TASKS]").setLevel(logging.INFO)


async def run_task(goal: str, context: dict | None = None, max_depth: int = DEFAULT_MAX_DEPTH) -> None:
    """创建根 Agent 并执行任务。"""
    # 初始化组件
    registry = create_default_registry()
    reviewer = CompositeReviewer(
        schema_reviewer=SchemaReviewer(),
        hardcode_reviewer=HardcodeRuleReviewer(),
        semantic_reviewer=LLMReviewer(),
    )

    agent = BaseAgent(
        role="根智能体",
        skills=registry,
        max_depth=max_depth,
        reviewer=reviewer,
    )

    task = TaskRequest(
        goal=goal,
        context=context or {},
    )

    print(f"\n🚀 启动任务: {goal}")
    print(f"   任务 ID: {task.task_id}")
    print(f"   最大递归深度: {max_depth}")
    print("-" * 60)

    try:
        result = await agent.solve(task)
        print("\n✅ 任务完成!")
        print(f"   状态: {result.status}")
        print(f"   结果: {json.dumps(result.data, ensure_ascii=False, indent=2)}")
        if result.artifacts:
            print(f"   产物: {result.artifacts}")
    except MaxRetryError as e:
        print(f"\n❌ 任务失败 (超出最大重试次数)")
        print(f"   任务 ID: {e.task_id}")
        print(f"   重试次数: {e.retries}")
        print(f"   最后反馈: {e.last_feedback}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 任务异常: {e}")
        logging.exception("Unexpected error")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="FAW - 递归 Map-Reduce 智能体网络",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例:\n"
            '  python main.py "计算 (3 + 5) * 2 的结果"\n'
            '  python main.py "分析并总结以下文本的关键观点" --context \'{"text": "..."}\'\n'
            '  python main.py --demo\n'
        ),
    )
    parser.add_argument("goal", nargs="?", help="任务目标描述")
    parser.add_argument(
        "--context",
        type=str,
        default="{}",
        help="JSON 格式的任务上下文 (默认: {})",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=DEFAULT_MAX_DEPTH,
        help=f"最大递归深度 (默认: {DEFAULT_MAX_DEPTH})",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="运行演示任务",
    )
    # 移除统一的 --debug
    parser.add_argument(
        "--debug-llm",
        action="store_true",
        help="开启 LLM API 详细请求/返回日志",
    )
    parser.add_argument(
        "--debug-tasks",
        action="store_true",
        help="开启任务流转与并发依赖图过程日志",
    )
    parser.add_argument(
        "--debug-skills",
        action="store_true",
        help="开启技能按需装载日志",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="指定日志记录到的文件路径 (例如: faw_run.log)",
    )

    args = parser.parse_args()

    # 此处利用命令行覆盖环境变量配置
    setup_logging(
        debug_llm=args.debug_llm,
        debug_tasks=args.debug_tasks,
        debug_skills=args.debug_skills,
        log_file=args.log_file,
    )
    logger = logging.getLogger(__name__)
    logger.info("Starting FAW main process...")

    if args.demo:
        goal = "计算 (15 + 27) * 3 - 10 的结果，然后将结果数字转换为大写文本。"
        context = {}
    elif args.goal:
        goal = args.goal
        try:
            context = json.loads(args.context)
        except json.JSONDecodeError:
            print(f"❌ 无法解析 context JSON: {args.context}")
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(0)

    asyncio.run(run_task(goal, context, args.max_depth))


if __name__ == "__main__":
    main()
