"""DeepSeek / OpenAI 兼容 LLM 工厂。"""

from __future__ import annotations

from functools import lru_cache

from langchain_openai import ChatOpenAI

from . import config


@lru_cache(maxsize=1)
def get_llm() -> ChatOpenAI:
    """返回 ChatOpenAI 客户端，指向 DeepSeek 的 OpenAI 兼容接口。"""
    if not config.DEEPSEEK_API_KEY:
        raise RuntimeError(
            "没有找到 DEEPSEEK_API_KEY。请复制 .env.example 为 .env，"
            "并填入 platform.deepseek.com 申请的 key。"
        )
    return ChatOpenAI(
        model=config.DEEPSEEK_MODEL,
        api_key=config.DEEPSEEK_API_KEY,
        base_url=config.DEEPSEEK_BASE_URL,
        temperature=0.0,
    )
