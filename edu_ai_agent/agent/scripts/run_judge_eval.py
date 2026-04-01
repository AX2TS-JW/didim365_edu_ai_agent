"""
2주차 3일차: LLM-as-a-Judge 평가 실행 스크립트

eval_dataset.json의 각 케이스에 대해:
1. 에이전트에 질문 → 답변 수집
2. LLM 판사가 1~5점 채점
3. 결과를 output/judge_report.json에 저장

사용법:
  cd edu_ai_agent/agent
  uv run python scripts/run_judge_eval.py

※ FastAPI 서버가 localhost:8000에서 실행 중이어야 합니다.
"""

import json
import sys
import uuid
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

# 프로젝트 루트를 path에 추가
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.evaluation.llm_judge import LLMJudge

AGENT_URL = "http://localhost:8000/api/v1/chat"
DATASET_FILE = Path(__file__).parent / "eval_dataset.json"
OUTPUT_FILE = Path(__file__).parent / "output" / "judge_report.json"


def call_agent(question: str) -> str:
    """에이전트에 질문을 보내고 최종 답변을 추출합니다."""
    thread_id = str(uuid.uuid4())
    content = ""
    with httpx.stream(
        "POST",
        AGENT_URL,
        json={"thread_id": thread_id, "message": question},
        timeout=120.0,
    ) as response:
        for line in response.iter_lines():
            line = line.strip()
            if line.startswith("data: "):
                try:
                    event = json.loads(line[6:])
                    if event.get("step") == "done" and event.get("content"):
                        content = event["content"]
                except json.JSONDecodeError:
                    continue
    return content


def main():
    with open(DATASET_FILE, encoding="utf-8") as f:
        dataset = json.load(f)

    print(f"=== LLM-as-a-Judge 평가 ({len(dataset)}건) ===\n")

    # 1. 에이전트에 질문 → 답변 수집
    print("📡 에이전트에 질문 전송 중...")
    answers = []
    for i, item in enumerate(dataset):
        q = item["input"]
        print(f"  [{i + 1}/{len(dataset)}] {q}")
        ans = call_agent(q)
        answers.append(ans)
        print(f"    → {len(ans)}자 수신")

    # 2. LLM 판사 채점
    print("\n⚖️  LLM 판사 채점 중 (gpt-4o-mini)...")
    judge = LLMJudge(model="gpt-4o-mini")

    judge_items = []
    for item, answer in zip(dataset, answers):
        judge_items.append({
            "question": item["input"],
            "answer": answer,
            "expected": item["expected_output"],
        })

    scores = judge.evaluate_batch(judge_items)

    # 3. 결과 합산
    results = []
    for item, answer, score_info in zip(dataset, answers, scores):
        results.append({
            "input": item["input"],
            "expected_tool": item.get("expected_tool"),
            "expected_region": item.get("expected_region"),
            "agent_answer": answer[:500],
            "judge_score": score_info["score"],
            "judge_reasoning": score_info["reasoning"],
        })

    total_score = sum(r["judge_score"] for r in results)
    avg_score = total_score / len(results) if results else 0

    report = {
        "summary": {
            "total_cases": len(results),
            "average_score": round(avg_score, 2),
            "total_score": total_score,
            "max_possible": len(results) * 5,
            "judge_model": "gpt-4o-mini",
        },
        "results": results,
    }

    # 4. 저장
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    # 5. 콘솔 출력
    print(f"\n{'═' * 60}")
    print(f"평균 점수: {avg_score:.2f} / 5.00")
    print(f"{'─' * 60}")
    for r in results:
        emoji = "🟢" if r["judge_score"] >= 4 else "🟡" if r["judge_score"] >= 3 else "🔴"
        print(f"  {emoji} [{r['judge_score']}/5] {r['input']}")
        print(f"      근거: {r['judge_reasoning'][:100]}")
    print(f"{'═' * 60}")
    print(f"리포트 저장: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
