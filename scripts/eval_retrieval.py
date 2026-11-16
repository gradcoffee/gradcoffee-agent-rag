"""检索评测：向量 / 混合 / 混合+rerank 的 top-3 命中对比。

python -m scripts.eval_retrieval
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag_agent.retrieval import get_retriever

DEFAULT_QUERIES = Path(__file__).resolve().parents[1] / "eval" / "retrieval_queries.json"


def hit(texts: list[str], expected: list[str]) -> bool:
    joined = "\n".join(texts).lower()
    return any(keyword.lower() in joined for keyword in expected)


def main() -> None:
    parser = argparse.ArgumentParser(description="检索 top-3 命中率对比")
    parser.add_argument("--queries", default=str(DEFAULT_QUERIES))
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--no-rerank", action="store_true")
    args = parser.parse_args()

    cases = json.loads(Path(args.queries).read_text(encoding="utf-8"))
    retriever = get_retriever()

    rows = []
    for case in cases:
        query = case["query"]
        expected = case["expected"]

        vector_ids = retriever.vector_search(query, k=args.top_k)
        vector_texts = [retriever.by_id[i].text for i in vector_ids if i in retriever.by_id]

        hybrid = retriever.retrieve(
            query, top_k=args.top_k, use_rerank=False
        )
        hybrid_texts = [item.text for item in hybrid]

        rerank_texts = []
        if not args.no_rerank:
            reranked = retriever.retrieve(query, top_k=args.top_k, use_rerank=True)
            rerank_texts = [item.text for item in reranked]

        rows.append(
            (
                query,
                hit(vector_texts, expected),
                hit(hybrid_texts, expected),
                hit(rerank_texts, expected) if rerank_texts else None,
            )
        )

    print(f"{'query':<32} vector  hybrid  rerank")
    print("-" * 62)
    for query, vector_hit, hybrid_hit, rerank_hit in rows:
        rerank_mark = "-" if rerank_hit is None else ("✓" if rerank_hit else "✗")
        print(
            f"{query[:30]:<32} {'✓' if vector_hit else '✗':<6} "
            f"{'✓' if hybrid_hit else '✗':<6} {rerank_mark}"
        )


if __name__ == "__main__":
    main()
