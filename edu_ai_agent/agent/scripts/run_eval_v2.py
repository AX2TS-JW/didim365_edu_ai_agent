"""
복잡한 데이터셋(v2) 평가 스크립트

단순 조회(3) + 비교 분석(3) + 추론(2) + 엣지 케이스(2) = 10건
에이전트에 질문 → 답변 수집 → Opik 트레이싱으로 내부 동작 기록
결과를 output/eval_v2_report.json에 저장

사용법:
  cd edu_ai_agent/agent
  uv run python scripts/run_eval_v2.py

※ FastAPI 서버가 localhost:8000에서 실행 중이어야 합니다.
"""

import json
import time
import uuid
from pathlib import Path

import httpx

AGENT_URL = "http://localhost:8000/api/v1/chat"
DATASET_FILE = Path(__file__).parent / "eval_dataset_v2.json"
OUTPUT_FILE = Path(__file__).parent / "output" / "eval_v2_report.json"


def call_agent(question: str) -> dict:
    """에이전트에 질문을 보내고 답변 + 도구 호출 정보를 캡처합니다."""
    thread_id = str(uuid.uuid4())
    final_content = ""
    tool_calls = []
    tool_contents = []

    start = time.time()
    try:
        with httpx.stream(
            "POST",
            AGENT_URL,
            json={"thread_id": thread_id, "message": question},
            timeout=120.0,
        ) as response:
            for line in response.iter_lines():
                line = line.strip()
                if not line.startswith("data: "):
                    continue
                try:
                    event = json.loads(line[6:])
                except json.JSONDecodeError:
                    continue

                step = event.get("step")
                if step == "model":
                    calls = event.get("tool_calls", [])
                    for c in calls:
                        tool_calls.append(c)
                elif step == "tools":
                    content = event.get("content", "")
                    name = event.get("name", "")
                    if isinstance(content, str) and len(content) > 0:
                        tool_contents.append({"name": name, "content": content[:300]})
                elif step == "done":
                    final_content = event.get("content", "")
    except Exception as e:
        final_content = f"[ERROR] {e}"

    elapsed = round(time.time() - start, 1)
    return {
        "final_answer": final_content,
        "tool_calls": tool_calls,
        "tool_contents": tool_contents,
        "elapsed_seconds": elapsed,
    }


def evaluate_result(case: dict, result: dict) -> dict:
    """결정론적 기본 검증 (LLM 없이)"""
    checks = {}

    # 1. 답변 존재 여부 (에러 메시지, 강제 종료 메시지는 유효 답변이 아님)
    answer = result["final_answer"]
    error_phrases = ["[ERROR]", "처리 중 오류가 발생했습니다", "조회 가능한 데이터를 모두 확인했으나"]
    checks["has_answer"] = len(answer) > 20 and not any(p in answer for p in error_phrases)

    # 2. 도구 선택 확인
    expected_tool = case.get("expected_tool")
    called = result["tool_calls"]
    if expected_tool is None:
        # 도구 호출 없이 되물어야 하는 케이스
        real_tools = [c for c in called if c not in ["Planning", "ChatResponse"]]
        checks["tool_selection"] = len(real_tools) == 0
        checks["tool_detail"] = f"expected=none, actual={called}"
    elif expected_tool == "both":
        has_trades = any("trades" in str(c).lower() for c in called)
        has_rentals = any("rentals" in str(c).lower() for c in called)
        checks["tool_selection"] = has_trades and has_rentals
        checks["tool_detail"] = f"expected=both, actual={called}, trades={has_trades}, rentals={has_rentals}"
    elif expected_tool == "both_trades":
        trade_calls = [c for c in called if "trades" in str(c).lower()]
        checks["tool_selection"] = len(trade_calls) >= 2
        checks["tool_detail"] = f"expected=2x trades, actual={called}, trade_calls={len(trade_calls)}"
    else:
        checks["tool_selection"] = any(expected_tool in str(c).lower() for c in called)
        checks["tool_detail"] = f"expected={expected_tool}, actual={called}"

    # 3. 응답 시간
    checks["response_time"] = result["elapsed_seconds"] <= 60

    # 4. Tool 호출 횟수 (ChatResponse, Planning 제외)
    real_tool_count = len([c for c in called if c not in ["Planning", "ChatResponse"]])
    checks["within_call_limit"] = real_tool_count <= 3
    checks["tool_count"] = real_tool_count

    return checks


def main():
    with open(DATASET_FILE, encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"=== 복잡 데이터셋 평가 ({len(dataset)}건) ===")
    print(f"카테고리: 단순 조회, 비교 분석, 추론, 엣지 케이스\n")

    results = []
    category_scores = {}

    for i, case in enumerate(dataset):
        cat = case.get("category", "")
        q = case["input"]
        print(f"  [{i+1}/{len(dataset)}] [{cat}] {q}")

        result = call_agent(q)
        checks = evaluate_result(case, result)

        passed = sum(1 for k, v in checks.items() if isinstance(v, bool) and v)
        total = sum(1 for k, v in checks.items() if isinstance(v, bool))
        score = passed / total if total > 0 else 0

        # 카테고리별 집계
        if cat not in category_scores:
            category_scores[cat] = {"total": 0, "score_sum": 0}
        category_scores[cat]["total"] += 1
        category_scores[cat]["score_sum"] += score

        emoji = "✅" if score >= 0.75 else "🟡" if score >= 0.5 else "❌"
        print(f"    {emoji} score={score:.0%} | {result['elapsed_seconds']}초 | tools={result['tool_calls']}")

        if not checks.get("tool_selection"):
            print(f"    ⚠️  도구 선택 불일치: {checks.get('tool_detail', '')}")
        if not checks.get("has_answer"):
            print(f"    ⚠️  답변 없음/에러: {result['final_answer'][:100]}")

        results.append({
            "id": case.get("id"),
            "category": cat,
            "input": q,
            "expected_tool": case.get("expected_tool"),
            "agent_answer": result["final_answer"][:500],
            "tool_calls": result["tool_calls"],
            "tool_contents_count": len(result["tool_contents"]),
            "elapsed_seconds": result["elapsed_seconds"],
            "checks": {k: v for k, v in checks.items() if isinstance(v, bool)},
            "score": round(score, 2),
        })

    # 요약
    print(f"\n{'═' * 60}")
    print("카테고리별 결과:")
    for cat, data in category_scores.items():
        avg = data["score_sum"] / data["total"]
        emoji = "✅" if avg >= 0.75 else "🟡" if avg >= 0.5 else "❌"
        print(f"  {emoji} {cat}: {avg:.0%} ({data['total']}건)")

    overall = sum(r["score"] for r in results) / len(results)
    print(f"\n전체 평균: {overall:.0%}")
    print(f"{'═' * 60}")

    # 저장
    report = {
        "summary": {
            "total_cases": len(results),
            "overall_score": round(overall, 3),
            "by_category": {
                cat: round(data["score_sum"] / data["total"], 3)
                for cat, data in category_scores.items()
            },
        },
        "results": results,
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\n리포트 저장: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
