"""混合检索：向量 + BM25 → RRF → 可选 rerank。"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path

import jieba
from langchain_community.vectorstores import FAISS
from rank_bm25 import BM25Okapi

from . import config
from .chunking import ChunkRecord
from .embeddings import get_embeddings
from .fusion import rrf_fuse

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RetrievedChunk:
    id: int
    text: str
    source: str
    title_path: str
    score: float
    method: str

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "source": self.source,
            "title_path": self.title_path,
            "score": self.score,
            "method": self.method,
        }


def _tokenize(text: str) -> list[str]:
    return [token.strip() for token in jieba.cut(text) if token.strip()]


class HybridRetriever:
    """从本地索引加载向量库和 BM25，先召回再精排。"""

    def __init__(
        self,
        index_dir: Path | str | None = None,
        *,
        enable_rerank: bool | None = None,
    ) -> None:
        self.index_dir = Path(index_dir) if index_dir else config.INDEX_DIR
        chunks_path = self.index_dir / "chunks.json"
        if not chunks_path.exists():
            raise FileNotFoundError(
                f"没有找到索引：{chunks_path}。先运行 "
                "`python -m scripts.ingest` 建索引。"
            )

        payload = json.loads(chunks_path.read_text(encoding="utf-8"))
        self.chunks = [ChunkRecord(**item) for item in payload["chunks"]]
        self.by_id = {chunk.id: chunk for chunk in self.chunks}
        self.corpus = [chunk.text for chunk in self.chunks]
        self.ids = [chunk.id for chunk in self.chunks]

        self.embeddings = get_embeddings()
        self.vector_store = FAISS.load_local(
            str(self.index_dir),
            self.embeddings,
            allow_dangerous_deserialization=True,
        )
        self.tokenized_corpus = [_tokenize(text) for text in self.corpus]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self.enable_rerank = (
            config.ENABLE_RERANK if enable_rerank is None else enable_rerank
        )
        self._reranker = None

    def vector_search(self, query: str, k: int = 20) -> list[int]:
        hits = self.vector_store.similarity_search_with_score(query, k=k)
        ids: list[int] = []
        for document, _score in hits:
            value = document.metadata.get("id")
            if value is not None:
                ids.append(int(value))
                continue
            # 兼容旧索引：没有 id 时按文本反查，重复文本会有歧义。
            for chunk in self.chunks:
                if chunk.text == document.page_content:
                    ids.append(chunk.id)
                    break
        return ids

    def bm25_search(self, query: str, k: int = 20) -> list[int]:
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = [float(score) for score in self.bm25.get_scores(tokens)]
        ranked = [
            index
            for index, score in enumerate(scores)
            if score > 0
        ]
        ranked.sort(key=lambda index: scores[index], reverse=True)
        return [self.ids[index] for index in ranked[:k]]

    def hybrid_search(self, query: str, k: int = 20) -> list[tuple[int, float]]:
        return rrf_fuse(
            self.vector_search(query, k=k),
            self.bm25_search(query, k=k),
            limit=k,
        )

    def _get_reranker(self):
        if self._reranker is None:
            from sentence_transformers import CrossEncoder

            self._reranker = CrossEncoder(config.RERANKER_MODEL)
        return self._reranker

    def _rerank(
        self,
        query: str,
        candidates: list[tuple[int, float]],
        top_k: int,
    ) -> list[tuple[int, float]]:
        pairs = [(query, self.by_id[chunk_id].text) for chunk_id, _ in candidates]
        raw_scores = self._get_reranker().predict(pairs)
        scored = [
            (candidate[0], float(score))
            for candidate, score in zip(candidates, raw_scores, strict=True)
        ]
        scored.sort(key=lambda item: (-item[1], item[0]))
        return scored[:top_k]

    def retrieve(
        self,
        query: str,
        *,
        recall_k: int | None = None,
        top_k: int | None = None,
        use_rerank: bool | None = None,
    ) -> list[RetrievedChunk]:
        """召回 top-20，按需 rerank 到 top-3。"""
        recall_k = recall_k or config.RECALL_K
        top_k = top_k or config.FINAL_K
        use_rerank = self.enable_rerank if use_rerank is None else use_rerank

        candidates = self.hybrid_search(query, k=recall_k)
        if not candidates:
            return []

        method = "hybrid"
        if use_rerank:
            try:
                candidates = self._rerank(query, candidates, top_k)
                method = "hybrid+rerank"
            except Exception as exc:  # 模型下载失败 / 显存不足时退化为 RRF
                logger.warning("rerank 失败，退回 RRF 排序：%s", exc)

        results: list[RetrievedChunk] = []
        for chunk_id, score in candidates[:top_k]:
            chunk = self.by_id.get(chunk_id)
            if chunk is None:
                continue
            results.append(
                RetrievedChunk(
                    id=chunk.id,
                    text=chunk.text,
                    source=chunk.source,
                    title_path=chunk.title_path,
                    score=score,
                    method=method,
                )
            )
        return results

    @staticmethod
    def format_context(results: list[RetrievedChunk]) -> str:
        """把检索结果拼成给模型看的上下文，带出处和 id。"""
        if not results:
            return "没有检索到相关资料。"
        blocks = []
        for item in results:
            blocks.append(
                f"[{item.id}] 来源：{item.title_path}（{item.source}）\n{item.text}"
            )
        return "\n\n---\n\n".join(blocks)


_RETRIEVER: HybridRetriever | None = None
_RETRIEVER_LOCK = threading.Lock()


def get_retriever(index_dir: Path | str | None = None) -> HybridRetriever:
    """进程内复用检索器，避免每轮重复加载模型。"""
    global _RETRIEVER
    if _RETRIEVER is None:
        with _RETRIEVER_LOCK:
            if _RETRIEVER is None:
                _RETRIEVER = HybridRetriever(index_dir)
    return _RETRIEVER


def reset_retriever() -> None:
    global _RETRIEVER
    with _RETRIEVER_LOCK:
        _RETRIEVER = None
