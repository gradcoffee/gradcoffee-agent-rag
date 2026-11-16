"""Pydantic DTO：结构化输出和 API 请求体。"""

from __future__ import annotations

from pydantic import BaseModel, Field


class AnswerWithSources(BaseModel):
    """RAG 回答：答案 + 引用来源 + 置信度。"""

    answer: str = Field(description="对用户问题的完整回答")
    sources: list[str] = Field(
        default_factory=list,
        description="回答引用的文档标题列表，没有引用则为空列表",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="对回答的置信度，0.0 到 1.0，不确定就给低分",
    )


class ContractInfo(BaseModel):
    """合同关键信息，用于结构化输出示例。"""

    party_a: str = Field(description="甲方公司全称")
    party_b: str = Field(description="乙方公司全称")
    amount: float = Field(description="合同金额，单位：元")
    sign_date: str = Field(description="签订日期，格式 YYYY-MM-DD")


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, description="用户问题")
    user_id: str = Field(default="demo_user", min_length=1, max_length=128)
    thread_id: str = Field(default="default", min_length=1, max_length=128)


class SearchRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=3, ge=1, le=20)
