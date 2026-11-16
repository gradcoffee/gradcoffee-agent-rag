"""命令行问答：python -m scripts.ask "入职半年年假有几天？" """

from __future__ import annotations

import argparse
import asyncio

from rag_agent.graph import ainvoke_answer, stream_text


async def run(args) -> None:
    if args.no_stream:
        answer = await ainvoke_answer(
            args.question,
            user_id=args.user,
            thread_id=args.thread,
        )
        print(answer)
        return

    async for token in stream_text(
        args.question,
        user_id=args.user,
        thread_id=args.thread,
    ):
        print(token, end="", flush=True)
    print()


def main() -> None:
    parser = argparse.ArgumentParser(description="和知识库 Agent 对话一次")
    parser.add_argument("question")
    parser.add_argument("--user", default="demo_user")
    parser.add_argument("--thread", default="default")
    parser.add_argument("--no-stream", action="store_true")
    args = parser.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
