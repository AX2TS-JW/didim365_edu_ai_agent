"""
2주차 4일차: DeepEval 메트릭 설정 모듈

DeepEval 라이브러리의 LLM 기반 메트릭을 에이전트 평가에 적용합니다.
- AnswerRelevancyMetric: 답변이 질문에 적절한가?
- FaithfulnessMetric: 답변이 도구 결과(context)에 충실한가?
- GEval: 커스텀 기준 평가

크레딧 절약을 위해 gpt-4o-mini를 기본 모델로 사용합니다.
"""

from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams


def create_relevancy_metric(threshold: float = 0.7) -> AnswerRelevancyMetric:
    """답변 적절성 메트릭을 생성합니다."""
    return AnswerRelevancyMetric(
        threshold=threshold,
        model="gpt-4o-mini",
    )


def create_faithfulness_metric(threshold: float = 0.7) -> FaithfulnessMetric:
    """답변 충실성 메트릭을 생성합니다. (도구 결과 대비 할루시네이션 여부)"""
    return FaithfulnessMetric(
        threshold=threshold,
        model="gpt-4o-mini",
    )


def create_tool_appropriateness_metric() -> GEval:
    """도구 활용 적절성을 평가하는 커스텀 G-Eval 메트릭"""
    return GEval(
        name="도구 활용 적절성",
        criteria=(
            "부동산 실거래가 에이전트가 사용자의 질문 의도에 맞는 도구를 선택했는지 평가합니다. "
            "매매 질문에는 매매 도구를, 전월세 질문에는 전월세 도구를 사용해야 합니다. "
            "지역과 시기도 질문에 부합해야 합니다."
        ),
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
        ],
        model="gpt-4o-mini",
    )


def create_test_case(
    question: str,
    agent_answer: str,
    expected_output: str = "",
    retrieval_context: list[str] | None = None,
) -> LLMTestCase:
    """DeepEval 테스트 케이스를 생성합니다."""
    return LLMTestCase(
        input=question,
        actual_output=agent_answer,
        expected_output=expected_output or None,
        retrieval_context=retrieval_context or [],
    )
