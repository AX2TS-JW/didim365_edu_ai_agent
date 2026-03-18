"""
2주차 과제: Opik Experiment 실행 스크립트

1. Dataset 생성 (부동산 Q&A 5개)
2. 에이전트에 질문 → 답변 수집 (순차)
3. AnswerRelevance 메트릭으로 자동 채점
4. jae-experiment-1 이름으로 Experiment 생성
"""

import os
import json
import uuid
import httpx
from dotenv import load_dotenv

load_dotenv()

# Opik 환경변수 설정
os.environ["OPIK_URL_OVERRIDE"] = os.getenv("OPIK__URL_OVERRIDE", "")
os.environ["OPIK_WORKSPACE"] = os.getenv("OPIK__WORKSPACE", "default")
os.environ["OPIK_PROJECT_NAME"] = os.getenv("OPIK__PROJECT", "jae-project")

from opik import Opik, evaluate
from opik.evaluation.metrics import AnswerRelevance

AGENT_URL = "http://localhost:8000/api/v1/chat"

# ── 1. Opik 클라이언트 + Dataset 생성 ──

client = Opik(
    host=os.getenv("OPIK__URL_OVERRIDE"),
    project_name="jae-project",
)

dataset = client.get_or_create_dataset(name="jae-dataset")

test_items = [
    {
        "input": "강남구 최근 아파트 매매 시세 알려줘",
        "expected_output": "강남구의 최근 아파트 매매 실거래가 데이터를 조회하여 주요 단지별 거래가격과 평당가를 안내합니다.",
    },
    {
        "input": "서초구 반포동 아파트 전세 시세 알려줘",
        "expected_output": "서초구 반포동의 최근 아파트 전세 실거래가를 조회하여 주요 단지별 전세가를 안내합니다.",
    },
    {
        "input": "분당구 아파트 매매 실거래가 조회해줘",
        "expected_output": "분당구의 최근 아파트 매매 실거래가를 조회하여 주요 단지별 거래 내역을 안내합니다.",
    },
    {
        "input": "역삼동 30평대 아파트 매매가 알려줘",
        "expected_output": "역삼동의 30평대(전용면적 약 84~99㎡) 아파트 매매 실거래가를 조회하여 안내합니다.",
    },
    {
        "input": "송파구 아파트 월세 시세 알려줘",
        "expected_output": "송파구의 최근 아파트 월세 실거래가를 조회하여 주요 단지별 보증금과 월세를 안내합니다.",
    },
]

dataset.insert(test_items)
print(f"Dataset 생성 완료: jae-dataset ({len(test_items)}건)")


# ── 2. 에이전트에 질문하고 답변 미리 수집 ──

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


print("\n에이전트에 질문을 순차적으로 보내는 중...")
answers = {}
for i, item in enumerate(test_items):
    q = item["input"]
    print(f"  [{i+1}/{len(test_items)}] {q}")
    ans = call_agent(q)
    answers[q] = ans
    print(f"         → 답변 수신 ({len(ans)}자)")

print(f"\n답변 수집 완료: {len(answers)}건")


# ── 3. 수집된 답변으로 Experiment 실행 ──

def evaluation_task(dataset_item: dict) -> dict:
    """미리 수집한 답변을 반환합니다."""
    question = dataset_item["input"]
    return {
        "input": question,
        "output": answers.get(question, "답변 없음"),
    }


print("\nExperiment 채점 시작: jae-experiment-1")

result = evaluate(
    dataset=dataset,
    task=evaluation_task,
    scoring_metrics=[
        AnswerRelevance(model="openai/gpt-4.1", require_context=False),
    ],
    experiment_name="jae-experiment-1",
    project_name="jae-project",
    task_threads=1,
)

print("\n=== Experiment 완료 ===")
print("Experiment: jae-experiment-1")
print(f"Dataset: jae-dataset ({len(test_items)}건)")
print("Opik 대시보드에서 결과를 확인하세요!")
