"""交互式命令行：python -m scripts.chat --user demo_user"""

from __future__ import annotations

import argparse
import asyncio

from rag_agent.graph import stream_text


async def run(args) -> None:
    print("输入问题，回车发送；输入 /quit 退出。\n")
    while True:
        try:
            question = input("你：").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not question:
            continue
        if question in {"/quit", "/exit"}:
            return

        print("Agent：", end="", flush=True)
        try:
            async for token in stream_text(
                question,
                user_id=args.user,
                thread_id=args.thread,
            ):
                print(token, end="", flush=True)
        except Exception as exc:
            print(f"\n[错误] {exc}")
        print("\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="交互式知识库 Agent")
    parser.add_argument("--user", default="demo_user")
    parser.add_argument("--thread", default="default")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
