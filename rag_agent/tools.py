"""Agent 工具：search_docs。"""

from __future__ import annotations

from langchain_core.tools import tool

from .retrieval import get_retriever


def make_search_tool():
    @tool
    def search_docs(query: str) -> str:
        """搜索内部知识库，回答公司流程、产品文档、技术规范相关问题。"""
        try:
            retriever = get_retriever()
            results = retriever.retrieve(query, top_k=3)
            if not results:
                return "没有检索到相关资料。"
            return retriever.format_context(results)
        except Exception as exc:
            return f"知识库检索失败：{exc}"

    return search_docs
