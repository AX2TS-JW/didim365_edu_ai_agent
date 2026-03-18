"""
2주차 4일차: ToolUsageMetric 단위 테스트

결정론적 메트릭이므로 LLM 호출 없이 테스트 가능합니다.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.evaluation.tool_usage_metric import ToolUsageMetric


@pytest.fixture
def metric():
    return ToolUsageMetric()


class TestToolUsageMetric:
    """ToolUsageMetric 테스트"""

    def test_correct_tool_selection(self, metric):
        case = {"expected_tool": "search_apartment_trades", "expected_region": "강남구"}
        trace = {
            "tool_calls": [
                {"step": "model", "tool": "search_apartment_trades"},
                {"step": "tools", "name": "search_apartment_trades"},
            ],
            "final_answer": "강남구 아파트 매매 실거래가입니다. 래미안 84㎡ 23억...",
            "elapsed_seconds": 5.0,
        }
        result = metric.evaluate(case, trace)
        assert result["pass"] is True
        assert result["score"] == 1.0

    def test_wrong_tool_selection(self, metric):
        case = {"expected_tool": "search_apartment_trades", "expected_region": "강남구"}
        trace = {
            "tool_calls": [
                {"step": "model", "tool": "search_apartment_rentals"},
            ],
            "final_answer": "강남구 전세 데이터입니다...",
            "elapsed_seconds": 3.0,
        }
        result = metric.evaluate(case, trace)
        tool_check = next(c for c in result["checks"] if c["name"] == "tool_selection")
        assert tool_check["pass"] is False

    def test_call_limit_exceeded(self, metric):
        case = {"expected_tool": "search_apartment_trades"}
        trace = {
            "tool_calls": [
                {"step": "model", "tool": "search_apartment_trades"},
                {"step": "model", "tool": "search_apartment_trades"},
                {"step": "model", "tool": "search_apartment_trades"},
                {"step": "model", "tool": "search_apartment_trades"},
            ],
            "final_answer": "데이터 조회 결과...",
            "elapsed_seconds": 10.0,
        }
        result = metric.evaluate(case, trace)
        limit_check = next(c for c in result["checks"] if c["name"] == "call_limit")
        assert limit_check["pass"] is False

    def test_both_tools_expected(self, metric):
        case = {"expected_tool": "both"}
        trace = {
            "tool_calls": [
                {"step": "model", "tool": "search_apartment_trades"},
                {"step": "model", "tool": "search_apartment_rentals"},
            ],
            "final_answer": "매매와 전세를 비교하면...",
            "elapsed_seconds": 8.0,
        }
        result = metric.evaluate(case, trace)
        tool_check = next(c for c in result["checks"] if c["name"] == "tool_selection")
        assert tool_check["pass"] is True

    def test_timeout_answer(self, metric):
        case = {"expected_tool": "search_apartment_trades"}
        trace = {
            "tool_calls": [{"step": "model", "tool": "search_apartment_trades"}],
            "final_answer": "[TIMEOUT]",
            "elapsed_seconds": 35.0,
        }
        result = metric.evaluate(case, trace)
        answer_check = next(c for c in result["checks"] if c["name"] == "has_valid_answer")
        assert answer_check["pass"] is False
        time_check = next(c for c in result["checks"] if c["name"] == "response_time")
        assert time_check["pass"] is False

    def test_no_expected_tool(self, metric):
        """expected_tool이 없으면 도구 선택 체크를 건너뜁니다."""
        case = {}
        trace = {
            "tool_calls": [],
            "final_answer": "어느 지역의 아파트 시세를 알고 싶으신가요?",
            "elapsed_seconds": 1.0,
        }
        result = metric.evaluate(case, trace)
        # tool_selection 체크가 없어야 함
        check_names = [c["name"] for c in result["checks"]]
        assert "tool_selection" not in check_names
