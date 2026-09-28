"""DeepSeek request planning, or an explicit offline rule-simulation demo."""

import argparse
import getpass
import json
import os
import sys
from contextlib import redirect_stdout

from staysim_agent.session import AgentSession, DEFAULT_DB


def main():
    parser = argparse.ArgumentParser(description="StaySim Agent：创建、查看、确认、补充和恢复持久任务")
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--request", help="描述一次模拟；调用 DeepSeek 解析，但不自动执行")
    choice.add_argument("--demo", action="store_true", help="创建 1 位老人、20 秒规则模拟计划，不调用模型，仍需确认")
    choice.add_argument("--show", metavar="TASK_ID", help="查看持久任务")
    choice.add_argument("--approve", metavar="TASK_ID", help="确认并执行当前计划")
    choice.add_argument("--reply", metavar="TASK_ID", help="补充或修改需求，使用 --text 提供内容")
    choice.add_argument("--cancel", metavar="TASK_ID", help="取消等待中的任务")
    choice.add_argument("--recover", metavar="TASK_ID", help="恢复异常中断的任务；不会盲目重复模拟")
    choice.add_argument("--ui", action="store_true", help="启动 StaySim Agent Gradio 界面")
    parser.add_argument("--text", help="用于 --reply 的补充内容")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="SQLite 检查点路径")
    parser.add_argument("--ask-key", action="store_true", help="在本机终端隐藏输入 DeepSeek Key，仅当前进程使用")
    args = parser.parse_args()
    if args.ask_key:
        if not (args.request or args.reply or args.recover or args.ui) or not sys.stdin.isatty():
            parser.error("--ask-key 用于交互终端中的 --request、--reply、--recover 或 --ui")
        os.environ["STAYSIM_AGENT_API_KEY"] = getpass.getpass("DeepSeek API Key (hidden): ")
    if args.text and not args.reply:
        parser.error("--text 必须和 --reply 一起使用")
    if args.ui:
        from staysim_agent.ui import launch_agent_app
        launch_agent_app(args.db, inbrowser=True)
        return 0
    try:
        # Simulation progress must not corrupt the machine-readable JSON on stdout.
        with redirect_stdout(sys.stderr), AgentSession(args.db) as session:
            if args.demo:
                result = session.start(request={
                    "scenario_file": "community_square.yaml", "engine_mode": "rule",
                    "duration": 20, "dt": 2.0, "seed": 42, "counts": {"买菜老人": 1},
                })
            elif args.request:
                result = session.start(user_input=args.request)
            elif args.show:
                result = session.show(args.show)
            elif args.recover:
                result = session.recover(args.recover)
            else:
                action = "approve" if args.approve else "reply" if args.reply else "cancel"
                result = session.resume(args.approve or args.reply or args.cancel, action=action, text=args.text)
    except (ValueError, RuntimeError) as exc:
        result = {"status": "blocked", "error": str(exc)}
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 1 if result["status"] in ("failed", "blocked") else 0


if __name__ == "__main__":
    # PowerShell pipelines and saved output use UTF-8 consistently.
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
