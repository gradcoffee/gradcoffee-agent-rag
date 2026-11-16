"""建索引：python -m scripts.ingest"""

from __future__ import annotations

import argparse
import json

from rag_agent.indexer import build_index


def main() -> None:
    parser = argparse.ArgumentParser(description="把 docs/ 下的 Markdown 建成 RAG 索引")
    parser.add_argument("--docs", default=None, help="文档目录，默认 docs/")
    parser.add_argument("--index", default=None, help="索引目录，默认 data/index/")
    parser.add_argument(
        "--no-rebuild",
        action="store_true",
        help="已有 chunks.json 时直接复用，不重新 embedding",
    )
    args = parser.parse_args()

    payload = build_index(args.docs, args.index, rebuild=not args.no_rebuild)
    print(
        json.dumps(
            {
                "count": payload["count"],
                "embedding_model": payload["embedding_model"],
                "created_at": payload["created_at"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
