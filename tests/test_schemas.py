import pytest
from pydantic import ValidationError

from rag_agent.schemas import AnswerWithSources


def test_answer_schema_accepts_valid_payload():
    result = AnswerWithSources(
        answer="一线城市 500 元。",
        sources=["员工手册 > 差旅与报销"],
        confidence=0.91,
    )
    assert result.confidence == 0.91
    assert result.sources == ["员工手册 > 差旅与报销"]


def test_answer_schema_rejects_out_of_range_confidence():
    with pytest.raises(ValidationError):
        AnswerWithSources(answer="x", sources=[], confidence=1.2)
