"""
2주차 3일차: LLM Judge 단위 테스트

LLM 호출을 mock하여 판사 모듈의 로직을 검증합니다.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.evaluation.llm_judge import LLMJudge, JudgeResult, JUDGE_RUBRIC


class TestJudgeResult:
    """JudgeResult Pydantic 모델 테스트"""

    def test_valid_score(self):
        result = JudgeResult(score=3, reasoning="보통 수준의 답변")
        assert result.score == 3
        assert result.reasoning == "보통 수준의 답변"

    def test_min_score(self):
        result = JudgeResult(score=1, reasoning="부적절한 답변")
        assert result.score == 1

    def test_max_score(self):
        result = JudgeResult(score=5, reasoning="완벽한 답변")
        assert result.score == 5

    def test_invalid_score_too_low(self):
        with pytest.raises(Exception):
            JudgeResult(score=0, reasoning="범위 초과")

    def test_invalid_score_too_high(self):
        with pytest.raises(Exception):
            JudgeResult(score=6, reasoning="범위 초과")


class TestLLMJudge:
    """LLMJudge 클래스 테스트"""

    @patch("app.evaluation.llm_judge.ChatOpenAI")
    def test_evaluate_returns_result(self, mock_chat):
        mock_structured = MagicMock()
        mock_structured.invoke.return_value = JudgeResult(
            score=4, reasoning="데이터 기반 답변이나 평당가 누락"
        )
        mock_chat.return_value.with_structured_output.return_value = mock_structured

        judge = LLMJudge(model="gpt-4o-mini")
        result = judge.evaluate(
            question="강남구 매매 시세",
            answer="강남구 아파트 매매 실거래가입니다...",
            expected="강남구 매매 실거래가 안내",
        )

        assert result.score == 4
        assert "평당가" in result.reasoning
        mock_structured.invoke.assert_called_once()

    @patch("app.evaluation.llm_judge.ChatOpenAI")
    def test_evaluate_batch(self, mock_chat):
        mock_structured = MagicMock()
        mock_structured.invoke.side_effect = [
            JudgeResult(score=5, reasoning="완벽"),
            JudgeResult(score=2, reasoning="부족"),
        ]
        mock_chat.return_value.with_structured_output.return_value = mock_structured

        judge = LLMJudge()
        results = judge.evaluate_batch([
            {"question": "Q1", "answer": "A1", "expected": "E1"},
            {"question": "Q2", "answer": "A2", "expected": "E2"},
        ])

        assert len(results) == 2
        assert results[0]["score"] == 5
        assert results[1]["score"] == 2

    def test_rubric_contains_key_criteria(self):
        assert "정확성" in JUDGE_RUBRIC
        assert "완전성" in JUDGE_RUBRIC
        assert "1점" in JUDGE_RUBRIC
        assert "5점" in JUDGE_RUBRIC
