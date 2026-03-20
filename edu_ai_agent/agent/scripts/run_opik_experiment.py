"""
Opik Experiment 실행 스크립트 (v2)

1. jae-dataset을 새 v2 데이터셋으로 교체
2. 에이전트에 질문 → 답변 + 도구 호출 정보 수집
3. Opik 메트릭으로 자동 채점:
   - AnswerRelevance: 답변 적절성
   - AgentToolCorrectnessJudge: 도구 호출 정확성
   - Hallucination: 할루시네이션 검사
4. jae-experiment-v2 이름으로 Experiment 생성

사용법:
  cd edu_ai_agent/agent
  uv run python scripts/run_opik_experiment.py

※ FastAPI 서버가 localhost:8000에서 실행 중이어야 합니다.
"""

import os
import json
import uuid
from pathlib import Path

import httpx
from dotenv import load_dotenv

load_dotenv()

# Opik 환경변수
os.environ["OPIK_URL_OVERRIDE"] = os.getenv("OPIK__URL_OVERRIDE", "")
os.environ["OPIK_WORKSPACE"] = os.getenv("OPIK__WORKSPACE", "default")
os.environ["OPIK_PROJECT_NAME"] = os.getenv("OPIK__PROJECT", "jae-project")

from opik import Opik, evaluate
from opik.evaluation.metrics import AnswerRelevance, AgentToolCorrectnessJudge, Hallucination

AGENT_URL = "http://localhost:8000/api/v1/chat"
DATASET_FILE = Path(__file__).parent / "eval_dataset_v2.json"

# ── 1. Opik 클라이언트 + Dataset 갱신 ──

client = Opik(
    host=os.getenv("OPIK__URL_OVERRIDE"),
    project_name="jae-project",
)

# 기존 데이터셋 삭제 후 재생성
DATASET_NAME = "jae-dataset"
try:
    existing = client.get_dataset(name=DATASET_NAME)
    if existing:
        client.delete_dataset(name=DATASET_NAME)
        print(f"기존 데이터셋 삭제: {DATASET_NAME}")
except Exception:
    pass

dataset = client.get_or_create_dataset(name=DATASET_NAME)

# v2 데이터셋 로드 및 삽입
with open(DATASET_FILE, encoding="utf-8") as f:
    eval_cases = json.load(f)

# Opik 데이터셋 형식으로 변환
opik_items = []
for case in eval_cases:
    opik_items.append({
        "input": case["input"],
        "expected_output": case["expected_output"],
        "expected_tool": case.get("expected_tool") or "none",
        "category": case.get("category", ""),
    })

dataset.insert(opik_items)
print(f"데이터셋 갱신: {DATASET_NAME} ({len(opik_items)}건)")


# ── 2. 에이전트에 질문 → 답변 + 도구 호출 정보 수집 ──

def call_agent(question: str) -> dict:
    """에이전트에 질문을 보내고 답변 + 도구 호출 transcript를 캡처합니다."""
    thread_id = str(uuid.uuid4())
    final_content = ""
    tool_transcript = []  # 도구 호출 과정을 텍스트로 기록
    tool_contents = []    # 도구 결과 (context용)

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
                        if c != "Planning":
                            tool_transcript.append(f"Agent selected tool: {c}")
                elif step == "tools":
                    name = event.get("name", "")
                    content = str(event.get("content", ""))[:500]
                    tool_transcript.append(f"Tool {name} returned: {content[:200]}")
                    if content:
                        tool_contents.append(content)
                elif step == "done":
                    final_content = event.get("content", "")
    except Exception as e:
        final_content = f"[ERROR] {e}"

    # 도구 호출 과정을 하나의 텍스트로 합침 (AgentToolCorrectnessJudge용)
    transcript = "\n".join(tool_transcript) if tool_transcript else "No tools were called."

    return {
        "answer": final_content,
        "transcript": transcript,
        "context": tool_contents,
    }


# 미리 답변 수집 (Experiment에서는 빠르게 반환하기 위해)
print("\n📡 에이전트에 질문 전송 중...")
answers = {}
for i, case in enumerate(eval_cases):
    q = case["input"]
    print(f"  [{i+1}/{len(eval_cases)}] [{case.get('category', '')}] {q}")
    result = call_agent(q)
    answers[q] = result
    status = "✅" if "[ERROR]" not in result["answer"] else "❌"
    print(f"    {status} 답변 {len(result['answer'])}자, 도구 호출 {len(result['transcript'].splitlines())}건")

print(f"\n답변 수집 완료: {len(answers)}건")


# ── 3. Experiment 실행 ──

def evaluation_task(dataset_item: dict) -> dict:
    """미리 수집한 답변을 반환합니다."""
    question = dataset_item["input"]
    result = answers.get(question, {"answer": "답변 없음", "transcript": "", "context": []})

    return {
        "input": question,
        "output": result["answer"],
        "context": result["context"],
        # AgentToolCorrectnessJudge가 사용할 도구 호출 기록
        "transcript": result["transcript"],
    }


print("\n🔬 Experiment 채점 시작: jae-experiment-v2")

# 메트릭 설정
scoring_metrics = [
    AnswerRelevance(
        model="openai/gpt-4o-mini",
        require_context=False,  # context 없이도 평가 가능
    ),
    AgentToolCorrectnessJudge(
        model="openai/gpt-4o-mini",
    ),
    Hallucination(
        model="openai/gpt-4o-mini",
    ),
]

result = evaluate(
    dataset=dataset,
    task=evaluation_task,
    scoring_metrics=scoring_metrics,
    experiment_name="jae-experiment-v2",
    project_name="jae-project",
    task_threads=1,  # 순차 실행 (API 부하 방지)
)

print("\n=== Experiment 완료 ===")
print("Experiment: jae-experiment-v2")
print(f"Dataset: {DATASET_NAME} ({len(eval_cases)}건)")
print("메트릭: AnswerRelevance, AgentToolCorrectness, Hallucination")
print("Opik 대시보드에서 결과를 확인하세요!")
