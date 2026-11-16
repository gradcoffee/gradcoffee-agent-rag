"""LangGraph Agent：工具调用 + SQLite checkpointer + 用户级 Store。"""

from __future__ import annotations

import atexit
import sqlite3
import threading
from collections.abc import AsyncIterator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from . import config
from .llm import get_llm
from .memory import get_store, make_memory_tools
from .tools import make_search_tool

_CONN: sqlite3.Connection | None = None
_CHECKPOINTER: SqliteSaver | None = None
_GRAPHS: dict[str, object] = {}
_CHECKPOINTER_LOCK = threading.Lock()
_GRAPH_LOCK = threading.Lock()


def _thread_key(user_id: str, thread_id: str) -> str:
    """把 user_id 和 thread_id 拼在一起，避免不同用户撞 thread。"""
    return f"{user_id}:{thread_id}"


def get_checkpointer() -> SqliteSaver:
    global _CONN, _CHECKPOINTER
    if _CHECKPOINTER is None:
        with _CHECKPOINTER_LOCK:
            if _CHECKPOINTER is None:
                config.CHECKPOINT_DB.parent.mkdir(parents=True, exist_ok=True)
                _CONN = sqlite3.connect(
                    str(config.CHECKPOINT_DB),
                    check_same_thread=False,
                )
                _CHECKPOINTER = SqliteSaver(_CONN)
                if hasattr(_CHECKPOINTER, "setup"):
                    _CHECKPOINTER.setup()
    return _CHECKPOINTER


def build_graph(user_id: str):
    """为一个用户编译 Agent。工具闭包里固定 user_id。"""
    tools = [make_search_tool(), *make_memory_tools(get_store(), user_id)]
    llm = get_llm().bind_tools(tools)

    async def call_model(state: MessagesState):
        messages = [SystemMessage(content=config.SYSTEM_PROMPT), *state["messages"]]
        response = await llm.ainvoke(messages)
        return {"messages": [response]}

    builder = StateGraph(MessagesState)
    builder.add_node("agent", call_model)
    builder.add_node("tools", ToolNode(tools))
    builder.add_edge(START, "agent")
    builder.add_conditional_edges("agent", tools_condition)
    builder.add_edge("tools", "agent")
    return builder.compile(
        checkpointer=get_checkpointer(),
        store=get_store(),
    )


def get_graph(user_id: str):
    if user_id not in _GRAPHS:
        with _GRAPH_LOCK:
            if user_id not in _GRAPHS:
                _GRAPHS[user_id] = build_graph(user_id)
    return _GRAPHS[user_id]


def _text_content(content) -> str:
    """兼容 str 和 LangChain 的 content block 列表。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, dict) and item.get("text"):
                parts.append(str(item["text"]))
            elif isinstance(item, str):
                parts.append(item)
        return "".join(parts)
    return "" if content is None else str(content)


async def stream_text(
    question: str,
    *,
    user_id: str = "demo_user",
    thread_id: str = "default",
) -> AsyncIterator[str]:
    """逐 token 输出；对应文章 04 的 chunked 文本流。"""
    graph = get_graph(user_id)
    configurable = {"configurable": {"thread_id": _thread_key(user_id, thread_id)}}
    async for chunk in graph.astream(
        {"messages": [HumanMessage(content=question)]},
        config=configurable,
        stream_mode="messages",
    ):
        message_chunk, _metadata = chunk
        if isinstance(message_chunk, AIMessage) or hasattr(message_chunk, "content"):
            text = _text_content(getattr(message_chunk, "content", ""))
            if text:
                yield text


async def ainvoke_answer(
    question: str,
    *,
    user_id: str = "demo_user",
    thread_id: str = "default",
) -> str:
    """非流式调用，返回最后一条消息的文本。"""
    graph = get_graph(user_id)
    configurable = {"configurable": {"thread_id": _thread_key(user_id, thread_id)}}
    result = await graph.ainvoke(
        {"messages": [HumanMessage(content=question)]},
        config=configurable,
    )
    return _text_content(result["messages"][-1].content)


def close_graph() -> None:
    global _CONN, _CHECKPOINTER, _GRAPHS
    _GRAPHS.clear()
    if _CONN is not None:
        try:
            _CONN.close()
        finally:
            _CONN = None
            _CHECKPOINTER = None


atexit.register(close_graph)
