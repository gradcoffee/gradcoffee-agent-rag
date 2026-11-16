"""建索引：Markdown → chunks → bge embedding → FAISS + chunks.json。"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document

from . import config
from .chunking import load_markdown_chunks
from .embeddings import get_embeddings


def build_index(
    docs_dir: Path | str | None = None,
    index_dir: Path | str | None = None,
    *,
    rebuild: bool = True,
) -> dict:
    """从 Markdown 文档构建 FAISS 索引，并写入 chunks.json。"""
    docs_root = Path(docs_dir) if docs_dir else config.DOCS_DIR
    index_root = Path(index_dir) if index_dir else config.INDEX_DIR
    index_root.mkdir(parents=True, exist_ok=True)

    if not rebuild and (index_root / "chunks.json").exists():
        return json.loads((index_root / "chunks.json").read_text(encoding="utf-8"))

    chunks = load_markdown_chunks(docs_root)
    documents = [
        Document(
            page_content=chunk.text,
            metadata={
                "id": chunk.id,
                "source": chunk.source,
                "title_path": chunk.title_path,
            },
        )
        for chunk in chunks
    ]

    vector_store = FAISS.from_documents(documents, get_embeddings())
    vector_store.save_local(str(index_root))

    payload = {
        "embedding_model": config.EMBEDDING_MODEL,
        "chunk_size": config.CHUNK_SIZE,
        "chunk_overlap": config.CHUNK_OVERLAP,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "count": len(chunks),
        "chunks": [chunk.to_dict() for chunk in chunks],
    }
    (index_root / "chunks.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload
