"""只看检索结果，不调 LLM：python -m scripts.search "delete_user 怎么加权限？" """

from __future__ import annotations

import argparse
import json

from rag_agent.retrieval import get_retriever


def main() -> None:
    parser = argparse.ArgumentParser(description="混合检索 + rerank 调试")
    parser.add_argument("query", help="检索问题")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--recall-k", type=int, default=20)
    parser.add_argument("--no-rerank", action="store_true", help="跳过交叉编码器精排")
    parser.add_argument("--json", action="store_true", help="输出 JSON")
    args = parser.parse_args()

    retriever = get_retriever()
    results = retriever.retrieve(
        args.query,
        recall_k=args.recall_k,
        top_k=args.top_k,
        use_rerank=not args.no_rerank,
    )

    if args.json:
        print(json.dumps([item.to_dict() for item in results], ensure_ascii=False, indent=2))
        return

    for item in results:
        print(f"score={item.score:.4f} id={item.id} [{item.method}] {item.title_path}")
        print(f"  {item.text[:160].replace(chr(10), ' ')}")
        print()


if __name__ == "__main__":
    main()
