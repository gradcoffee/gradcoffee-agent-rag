"""纯函数：RRF 融合。

不依赖 LangChain、FAISS 或模型，方便单测，也方便复用到别的检索器。
"""

from __future__ import annotations

from collections.abc import Sequence


def rrf_fuse(
    vector_ids: Sequence[int],
    bm25_ids: Sequence[int],
    *,
    limit: int = 20,
    k: int = 60,
) -> list[tuple[int, float]]:
    """按 Reciprocal Rank Fusion 合并两路召回结果。

    参数：
        vector_ids: 向量召回的 chunk id，按相关性从高到低。
        bm25_ids: BM25 召回的 chunk id，按相关性从高到低。
        limit: 最多返回多少个候选。
        k: RRF 平滑常数，常用 60。

    返回：
        [(chunk_id, rrf_score), ...]，按分数从高到低。
    """
    if limit <= 0:
        return []

    scores: dict[int, float] = {}
    for rank, chunk_id in enumerate(vector_ids):
        cid = int(chunk_id)
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
    for rank, chunk_id in enumerate(bm25_ids):
        cid = int(chunk_id)
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)

    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    return ordered[:limit]
