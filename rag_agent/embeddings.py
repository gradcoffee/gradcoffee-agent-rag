"""Embedding 模型工厂。"""

from __future__ import annotations

from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings

from . import config


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    """bge-small-zh 本地 embedding，向量做 L2 归一化。

    建索引和查询必须使用同一个模型；换模型就要重建索引。
    """
    return HuggingFaceEmbeddings(
        model_name=config.EMBEDDING_MODEL,
        encode_kwargs={"normalize_embeddings": True},
    )
