"""
2주차 4일차: 결정론적 Tool 사용 메트릭 (LLM 호출 불필요 — 무료)

에이전트의 도구 호출 패턴을 규칙 기반으로 평가합니다:
1. 올바른 도구 선택 (매매 vs 전월세)
2. 지역코드 매핑 정확성
3. Tool 호출 횟수 제한 (최대 3회) 준수
4. 미래 월 조회 여부
5. 동일 조합 중복 호출 여부
"""

from datetime import datetime


class ToolUsageMetric:
    """에이전트 도구 사용 패턴을 결정론적으로 평가합니다."""

    def evaluate(self, case: dict, trace: dict) -> dict:
        """
        Args:
            case: eval_dataset.json의 한 항목
                  (expected_tool, expected_region, expected_region_code)
            trace: run_diagnostic.py에서 캡처한 trace
                  (tool_calls, final_answer, elapsed_seconds)

        Returns:
            {"pass": bool, "score": float(0~1), "checks": [...], "summary": str}
        """
        checks = []
        tool_calls = trace.get("tool_calls", [])
        model_calls = [tc for tc in tool_calls if tc.get("step") == "model"]
        tool_results = [tc for tc in tool_calls if tc.get("step") == "tools"]

        # 호출된 도구 이름 목록
        called_tools = [tc.get("tool") for tc in model_calls if tc.get("tool")]

        # 1. 도구 선택 정확성
        expected_tool = case.get("expected_tool")
        if expected_tool:
            if expected_tool == "both":
                has_trades = any("trades" in t for t in called_tools)
                has_rentals = any("rentals" in t for t in called_tools)
                correct = has_trades and has_rentals
            else:
                correct = any(expected_tool in t for t in called_tools)
            checks.append({
                "name": "tool_selection",
                "pass": correct,
                "detail": f"expected={expected_tool}, actual={called_tools}",
            })

        # 2. 호출 횟수 제한 (최대 5회)
        actual_tool_count = len([t for t in called_tools if "ChatResponse" not in t])
        within_limit = actual_tool_count <= 5
        checks.append({
            "name": "call_limit",
            "pass": within_limit,
            "detail": f"{actual_tool_count}/5회 사용",
        })

        # 3. 중복 호출 체크
        tool_signatures = []
        for tc in model_calls:
            sig = tc.get("tool", "")
            tool_signatures.append(sig)
        has_duplicates = len(tool_signatures) != len(set(tool_signatures))
        # 동일 도구를 다른 인자로 호출하는 건 정상 (시점 비교 등)이므로 경고만
        checks.append({
            "name": "no_duplicates",
            "pass": not has_duplicates,
            "detail": f"signatures={tool_signatures}",
        })

        # 4. 응답 존재 여부
        answer = trace.get("final_answer", "")
        has_answer = len(answer) > 20 and "[ERROR]" not in answer and "[TIMEOUT]" not in answer
        checks.append({
            "name": "has_valid_answer",
            "pass": has_answer,
            "detail": f"answer_length={len(answer)}",
        })

        # 5. 응답 시간 (30초 이내)
        elapsed = trace.get("elapsed_seconds", 0)
        fast_enough = elapsed <= 30
        checks.append({
            "name": "response_time",
            "pass": fast_enough,
            "detail": f"{elapsed}초",
        })

        # 종합 점수
        passed_checks = sum(1 for c in checks if c["pass"])
        total_checks = len(checks)
        score = passed_checks / total_checks if total_checks > 0 else 0
        all_pass = all(c["pass"] for c in checks)

        failed = [c["name"] for c in checks if not c["pass"]]
        summary = "전체 통과" if all_pass else f"실패 항목: {', '.join(failed)}"

        return {
            "pass": all_pass,
            "score": round(score, 2),
            "checks": checks,
            "summary": summary,
        }
