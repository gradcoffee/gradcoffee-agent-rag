"""长期记忆：按 user_id 隔离的 Store，以及两个 Agent 工具。"""

from __future__ import annotations

import re
import threading

from langchain_core.tools import tool
from langgraph.store.memory import InMemoryStore

from . import config

_KEY_PATTERN = re.compile(r"[A-Za-z0-9_.-]{1,64}")
_STORE: InMemoryStore | None = None
_STORE_LOCK = threading.Lock()


def get_store() -> InMemoryStore:
    global _STORE
    if _STORE is None:
        with _STORE_LOCK:
            if _STORE is None:
                _STORE = InMemoryStore()
    return _STORE


def _namespace(user_id: str) -> tuple[str]:
    if not user_id or len(user_id) > 128:
        raise ValueError("user_id 不能为空，且长度不能超过 128")
    return (user_id,)


def _format_items(items) -> str:
    if not items:
        return "暂无相关记忆"
    lines = []
    for item in items:
        key = getattr(item, "key", "?")
        value = getattr(item, "value", {}) or {}
        content = value.get("content", str(value)) if isinstance(value, dict) else str(value)
        lines.append(f"- {key}：{content}")
    return "\n".join(lines)


def make_memory_tools(store, user_id: str) -> list:
    """为某个用户生成工具；user_id 通过闭包注入，不进 JSON Schema。"""

    namespace = _namespace(user_id)

    @tool
    def remember_fact(key: str, content: str) -> str:
        """记住关于当前用户的一个事实，例如偏好、过敏、工作习惯。"""
        if not _KEY_PATTERN.fullmatch(key):
            return "key 只能使用字母、数字、下划线、点或短横线，长度 1–64"
        if not content.strip():
            return "记忆内容不能为空"
        store.put(namespace=namespace, key=key, value={"content": content.strip()})
        return f"已记住：{content.strip()}"

    @tool
    def recall_facts() -> str:
        """查询当前用户的所有长期记忆；回答个性化问题前先调用。"""
        items = store.search(namespace, limit=config.MAX_MEMORY_ITEMS)
        return _format_items(items)

    return [remember_fact, recall_facts]


def list_memories(user_id: str) -> list[dict]:
    """给 API / 调试用的记忆快照。"""
    store = get_store()
    items = store.search(_namespace(user_id), limit=config.MAX_MEMORY_ITEMS)
    result = []
    for item in items:
        value = getattr(item, "value", {}) or {}
        content = value.get("content", str(value)) if isinstance(value, dict) else str(value)
        result.append(
            {
                "key": getattr(item, "key", "?"),
                "content": content,
            }
        )
    return result


def delete_memory(user_id: str, key: str) -> bool:
    """删除一条记忆；用户说“忘了我”时使用。"""
    if not _KEY_PATTERN.fullmatch(key):
        raise ValueError("非法 key")
    store = get_store()
    store.delete(namespace=_namespace(user_id), key=key)
    return True
