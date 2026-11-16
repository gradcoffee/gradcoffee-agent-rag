"""文档切块：按 Markdown 标题层级切，再按长度细分。

对应文章 05 的“按结构切，别硬切”。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from . import config


@dataclass(slots=True)
class ChunkRecord:
    id: int
    text: str
    source: str
    title_path: str

    def to_dict(self) -> dict:
        return asdict(self)


def _title_path(metadata: dict) -> str:
    parts = [
        str(metadata[key]).strip()
        for key in ("h1", "h2", "h3")
        if metadata.get(key)
    ]
    return " > ".join(parts)


def load_markdown_chunks(
    docs_dir: Path | str | None = None,
    *,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[ChunkRecord]:
    """读取 docs_dir 下所有 Markdown，返回带标题路径的块列表。"""
    root = Path(docs_dir) if docs_dir else config.DOCS_DIR
    if not root.exists():
        raise FileNotFoundError(f"文档目录不存在：{root}")

    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=False,
    )
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or config.CHUNK_SIZE,
        chunk_overlap=chunk_overlap or config.CHUNK_OVERLAP,
    )

    records: list[ChunkRecord] = []
    for path in sorted(root.rglob("*.md")):
        text = path.read_text(encoding="utf-8")
        if not text.strip():
            continue

        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            relative = path.name

        header_docs = md_splitter.split_text(text)
        for doc in header_docs:
            doc.metadata["source"] = relative
            doc.metadata["title_path"] = _title_path(doc.metadata) or path.stem

        for doc in splitter.split_documents(header_docs):
            content = doc.page_content.strip()
            if not content:
                continue
            records.append(
                ChunkRecord(
                    id=len(records),
                    text=content,
                    source=str(doc.metadata.get("source", relative)),
                    title_path=str(doc.metadata.get("title_path", path.stem)),
                )
            )

    if not records:
        raise ValueError(f"{root} 下没有可用的 Markdown 内容")
    return records
