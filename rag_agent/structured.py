"""Pydantic 结构化输出：AnswerWithSources 和 ContractInfo。"""

from __future__ import annotations

from collections.abc import Sequence

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, ValidationError

from .llm import get_llm
from .retrieval import RetrievedChunk
from .schemas import AnswerWithSources, ContractInfo


def invoke_with_retry(
    structured_llm,
    prompt: str,
    schema: type[BaseModel],
    *,
    max_retries: int = 2,
) -> BaseModel:
    """把 ValidationError 喂回模型，最多重试 max_retries 次。"""
    current_prompt = prompt
    last_error: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            result = structured_llm.invoke([HumanMessage(content=current_prompt)])
            if isinstance(result, schema):
                return result
            return schema.model_validate(result)
        except ValidationError as exc:
            last_error = exc
            if attempt == max_retries:
                break
            current_prompt = (
                f"上次你的输出格式不对：{exc}\n"
                "请严格按照字段要求重新输出，数值类型不要写中文。\n\n"
                f"原始问题：{prompt}"
            )

    raise last_error or RuntimeError("结构化输出失败")


def structured_answer(
    question: str,
    docs: Sequence[RetrievedChunk | str],
    *,
    max_retries: int = 2,
) -> AnswerWithSources:
    """基于检索资料，输出 answer / sources / confidence。"""
    if not docs:
        return AnswerWithSources(
            answer="资料里没有找到相关内容。",
            sources=[],
            confidence=0.0,
        )

    context_parts = []
    source_names: list[str] = []
    for item in docs:
        if isinstance(item, RetrievedChunk):
            context_parts.append(
                f"来源：{item.title_path}（{item.source}）\n{item.text}"
            )
            if item.title_path not in source_names:
                source_names.append(item.title_path)
        else:
            context_parts.append(str(item))
    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""根据以下资料回答用户问题。资料里没有的就说不知道，不要瞎编。

参考资料：
{context}

用户问题：{question}
"""
    structured_llm = get_llm().with_structured_output(
        AnswerWithSources,
        method="function_calling",
    )
    result = invoke_with_retry(
        structured_llm,
        prompt,
        AnswerWithSources,
        max_retries=max_retries,
    )
    if source_names and not result.sources:
        result = result.model_copy(update={"sources": source_names})
    return result


def extract_contract(
    contract_text: str,
    *,
    max_retries: int = 2,
) -> ContractInfo:
    """合同字段抽取示例，对应文章 03 的 ContractInfo。"""
    prompt = f"""从下面的合同文本中提取甲乙方、金额和签订日期。
金额统一换算成元；找不到的字段不要编造，留空字符串或 0。

合同文本：
{contract_text}
"""
    structured_llm = get_llm().with_structured_output(
        ContractInfo,
        method="function_calling",
    )
    return invoke_with_retry(
        structured_llm,
        prompt,
        ContractInfo,
        max_retries=max_retries,
    )
