"""
2주차 4일차: DeepEval 평가 실행 스크립트

eval_dataset.json의 각 케이스에 대해:
1. 에이전트에 질문 → 답변 + 도구 결과 수집
2. DeepEval 메트릭 (AnswerRelevancy, Faithfulness, G-Eval) 평가
3. ToolUsageMetric (결정론적) 평가
4. 결과를 output/deepeval_report.json에 저장

사용법:
  cd edu_ai_agent/agent
  uv run python scripts/run_deepeval.py

※ FastAPI 서버가 localhost:8000에서 실행 중이어야 합니다.
"""

import json
import sys
import uuid
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.evaluation.deepeval_metrics import (
    create_relevancy_metric,
    create_faithfulness_metric,
    create_tool_appropriateness_metric,
    create_test_case,
)
from app.evaluation.tool_usage_metric import ToolUsageMetric
from deepeval import evaluate as deepeval_evaluate

AGENT_URL = "http://localhost:8000/api/v1/chat"
DATASET_FILE = Path(__file__).parent / "eval_dataset_v2.json"
OUTPUT_FILE = Path(__file__).parent / "output" / "deepeval_report.json"


def call_agent_with_context(question: str) -> dict:
    """에이전트에 질문을 보내고 최종 답변 + 도구 결과(context)를 캡처합니다."""
    thread_id = str(uuid.uuid4())
    final_content = ""
    tool_contents = []
    tool_calls = []

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
                    for call in calls:
                        tool_calls.append({"step": "model", "tool": call})
                elif step == "tools":
                    content = event.get("content", "")
                    if isinstance(content, str) and len(content) > 0:
                        tool_contents.append(content)
                    tool_calls.append({
                        "step": "tools",
                        "name": event.get("name"),
                    })
                elif step == "done":
                    final_content = event.get("content", "")
    except Exception as e:
        final_content = f"[ERROR] {e}"

    return {
        "final_answer": final_content,
        "retrieval_context": tool_contents,
        "tool_calls": tool_calls,
    }


def main():
    with open(DATASET_FILE, encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"=== DeepEval 평가 ({len(dataset)}건) ===\n")

    # 1. 에이전트 답변 수집
    print("📡 에이전트에 질문 전송 중...")
    agent_results = []
    for i, item in enumerate(dataset):
        q = item["input"]
        print(f"  [{i + 1}/{len(dataset)}] {q}")
        result = call_agent_with_context(q)
        agent_results.append(result)
        print(f"    → 답변 {len(result['final_answer'])}자, context {len(result['retrieval_context'])}건")

    # 2. DeepEval 테스트 케이스 + 메트릭 생성
    print("\n📊 DeepEval 메트릭 평가 중 (gpt-4o-mini)...")
    relevancy = create_relevancy_metric()
    faithfulness = create_faithfulness_metric()
    tool_appropriateness = create_tool_appropriateness_metric()

    test_cases = []
    for item, result in zip(dataset, agent_results):
        tc = create_test_case(
            question=item["input"],
            agent_answer=result["final_answer"],
            expected_output=item.get("expected_output", ""),
            retrieval_context=result["retrieval_context"],
        )
        test_cases.append(tc)

    # DeepEval 일괄 평가
    metrics = [relevancy, faithfulness, tool_appropriateness]
    eval_result = deepeval_evaluate(
        test_cases=test_cases,
        metrics=metrics,
    )

    # 3. ToolUsageMetric (결정론적) 평가
    print("\n🔧 Tool 사용 메트릭 평가 중...")
    tool_metric = ToolUsageMetric()
    tool_results = []
    for item, result in zip(dataset, agent_results):
        tr = tool_metric.evaluate(item, result)
        tool_results.append(tr)

    # 4. 결과 합산
    results = []
    for i, (item, result, tool_r) in enumerate(zip(dataset, agent_results, tool_results)):
        tc = test_cases[i]
        entry = {
            "input": item["input"],
            "agent_answer": result["final_answer"][:500],
            "deepeval_metrics": {},
            "tool_usage": tool_r,
        }

        # DeepEval 메트릭 결과 추출
        for metric in metrics:
            try:
                metric.measure(tc)
                entry["deepeval_metrics"][metric.__class__.__name__] = {
                    "score": metric.score,
                    "passed": metric.is_successful(),
                    "reason": getattr(metric, "reason", ""),
                }
            except Exception as e:
                entry["deepeval_metrics"][metric.__class__.__name__] = {
                    "score": None,
                    "passed": False,
                    "reason": str(e),
                }

        results.append(entry)

    # 5. 요약 계산
    avg_scores = {}
    for metric_name in ["AnswerRelevancyMetric", "FaithfulnessMetric", "도구 활용 적절성"]:
        scores = [
            r["deepeval_metrics"].get(metric_name, {}).get("score")
            for r in results
            if r["deepeval_metrics"].get(metric_name, {}).get("score") is not None
        ]
        avg_scores[metric_name] = round(sum(scores) / len(scores), 3) if scores else None

    tool_avg = sum(r["tool_usage"]["score"] for r in results) / len(results) if results else 0

    report = {
        "summary": {
            "total_cases": len(results),
            "deepeval_averages": avg_scores,
            "tool_usage_avg_score": round(tool_avg, 3),
            "eval_model": "gpt-4o-mini",
        },
        "results": results,
    }

    # 6. 저장
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2, default=str)

    # 7. 콘솔 출력
    print(f"\n{'═' * 60}")
    print("DeepEval 평균 점수:")
    for name, score in avg_scores.items():
        print(f"  {name}: {score}")
    print(f"  ToolUsageMetric: {tool_avg:.3f}")
    print(f"{'─' * 60}")
    for r in results:
        tool_pass = "✅" if r["tool_usage"]["pass"] else "❌"
        print(f"  {tool_pass} {r['input']}")
        for mn, mv in r["deepeval_metrics"].items():
            s = mv.get("score", "N/A")
            print(f"      {mn}: {s}")
        print(f"      ToolUsage: {r['tool_usage']['summary']}")
    print(f"{'═' * 60}")
    print(f"리포트 저장: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
