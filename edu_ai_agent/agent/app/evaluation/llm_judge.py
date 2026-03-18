"""
2주차 3일차: LLM-as-a-Judge 평가 모듈

GPT-4o-mini를 판사로 설정하여 에이전트 답변의 정확성을 1~5점으로 채점합니다.
구조화 출력(Pydantic)으로 점수와 근거를 파싱합니다.
"""

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

JUDGE_RUBRIC = """당신은 부동산 실거래가 AI 에이전트의 답변 품질을 평가하는 심사위원입니다.

## 채점 기준 (1~5점)
- 1점: 완전히 틀리거나 무관한 답변
- 2점: 질문을 일부 다루지만 중대한 오류 포함
- 3점: 질문을 다루지만 정확성이나 세부 사항 부족
- 4점: 대체로 정확하나 사소한 문제 있음
- 5점: 정확하고 데이터 기반의 구조화된 완벽한 답변

## 평가 관점
1. **정확성**: 실거래가 데이터에 기반한 답변인가?
2. **완전성**: 가격, 면적, 층수, 날짜 등 핵심 정보가 포함되었는가?
3. **적절성**: 질문의 의도에 맞는 답변인가? (매매/전세 구분, 지역 정확성)
4. **구조**: 읽기 쉽게 정리되었는가?

## 입력
- 사용자 질문: {question}
- 에이전트 답변: {answer}
- 기대 답변 방향: {expected}

위 기준에 따라 점수와 근거를 제공하세요."""


class JudgeResult(BaseModel):
    """LLM 판사의 채점 결과"""
    score: int = Field(description="1~5점 채점 결과", ge=1, le=5)
    reasoning: str = Field(description="채점 근거 (한국어)")


class LLMJudge:
    """LLM 기반 답변 품질 판사"""

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0):
        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.structured_llm = self.llm.with_structured_output(JudgeResult)

    def evaluate(self, question: str, answer: str, expected: str = "") -> JudgeResult:
        """에이전트 답변을 채점합니다."""
        prompt = JUDGE_RUBRIC.format(
            question=question,
            answer=answer,
            expected=expected or "명시된 기대 답변 없음",
        )
        return self.structured_llm.invoke(prompt)

    def evaluate_batch(self, items: list[dict]) -> list[dict]:
        """여러 건을 일괄 채점합니다.

        Args:
            items: [{"question": ..., "answer": ..., "expected": ...}, ...]

        Returns:
            [{"question": ..., "answer": ..., "score": int, "reasoning": str}, ...]
        """
        results = []
        for item in items:
            result = self.evaluate(
                question=item["question"],
                answer=item["answer"],
                expected=item.get("expected", ""),
            )
            results.append({
                "question": item["question"],
                "answer": item["answer"][:200],
                "score": result.score,
                "reasoning": result.reasoning,
            })
        return results
