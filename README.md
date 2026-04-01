# 부동산 실거래가 분석 에이전트

자연어로 부동산 질문을 하면 **공공데이터(data.go.kr) API**에서 실거래가 데이터를 조회하고, LLM이 분석하여 답변하는 AI 에이전트입니다.

> "강남구 2025년 1월 아파트 매매가 알려줘" → 실거래 내역 조회 + 평당가 계산 + 요약 답변

---

## 목차

| # | 챕터 | 설명 |
|---|------|------|
| 1 | [프로젝트 개요](#1-프로젝트-개요) | 무엇을, 왜 만들었는가 |
| 2 | [아키텍처](#2-아키텍처) | 전체 시스템 구조와 통신 흐름 |
| 3 | [기술 스택](#3-기술-스택) | 사용된 기술과 선택 이유 |
| 4 | [에이전트 설계](#4-에이전트-설계) | LangChain 기반 ReAct 에이전트 구조 |
| 5 | [빠른 시작](#5-빠른-시작) | 로컬 환경 세팅 및 실행 |
| 6 | [사용 예시](#6-사용-예시) | 지원하는 질문 유형과 데모 |
| 7 | [LangChain 문서 매핑](#7-langchain-문서-매핑) | 프로젝트 코드 ↔ 공식 문서 대응표 |
| 8 | [관련 문서](#8-관련-문서) | 하위 문서 링크 |

---

## 1. 프로젝트 개요

### 배경

부동산 실거래가 데이터는 공공데이터포털에서 제공되지만, API 호출에 지역코드·날짜 파라미터가 필요하고 XML 응답을 직접 파싱해야 합니다. 이 프로젝트는 **자연어 한 문장**으로 이 과정을 자동화합니다.

### 핵심 기능

- **자연어 → API 호출**: "송파구 매매가 알려줘" → 지역코드 변환 → data.go.kr API 호출
- **매매 / 전월세 구분**: 질문 의도에 따라 적절한 Tool 자동 선택
- **멀티턴 대화**: 이전 맥락을 기억하여 후속 질문 처리
- **실시간 스트리밍**: SSE를 통한 단계별 응답 (추론 → 도구 호출 → 최종 답변)

---

## 2. 아키텍처

```
┌─────────────┐     POST /api/v1/chat      ┌──────────────┐
│  React UI   │  ─────────────────────────► │   FastAPI     │
│  (Vite)     │                             │  (Uvicorn)    │
│             │  ◄── SSE 스트리밍 ────────── │              │
│  :5173      │   {step, content, metadata}  │  :8000       │
└─────────────┘                             └──────┬───────┘
                                                   │
                                           create_agent()
                                                   │
                                           ┌───────▼───────┐
                                           │  LangChain     │
                                           │  Agent (ReAct)  │
                                           ├────────────────┤
                                           │ Tools:          │
                                           │ • 매매 실거래가  │──► data.go.kr API
                                           │ • 전월세 실거래가 │──► data.go.kr API
                                           ├────────────────┤
                                           │ Memory:         │
                                           │ InMemorySaver   │
                                           └────────────────┘
```

### 통신 흐름

```
사용자 입력 → POST /api/v1/chat (thread_id + message)
         → AgentService → LangChain Agent
         → ReAct 루프: 추론 → Tool 선택 → API 호출 → 응답 생성
         → SSE 스트리밍: model → tools → done
         → 프론트엔드 렌더링
```

---

## 3. 기술 스택

| 계층 | 기술 | 역할 |
|------|------|------|
| **Frontend** | React 19, TypeScript, Vite 7, MUI 6, Jotai | 채팅 UI + 상태 관리 |
| **Backend** | FastAPI, Uvicorn, Python 3.11~3.13 | API 서버 + SSE 스트리밍 |
| **AI Agent** | LangChain 1.x, LangGraph, OpenAI GPT-4.1 | ReAct 에이전트 + 도구 호출 |
| **데이터** | data.go.kr 공공데이터 API | 아파트 매매/전월세 실거래가 |
| **패키지 관리** | uv (Backend), pnpm (Frontend) | 의존성 관리 |

---

## 4. 에이전트 설계

### 프로젝트 구조

```
edu_ai_agent/
├── agent/                              # Backend
│   ├── app/
│   │   ├── agents/
│   │   │   ├── real_estate_agent.py    # create_agent() 에이전트 조립
│   │   │   ├── tools.py                # @tool 매매/전월세/전세가율 조회 + 지역명 검증
│   │   │   └── prompts.py             # get_system_prompt() 동적 프롬프트
│   │   ├── evaluation/                 # 2주차 평가 체계
│   │   │   ├── llm_judge.py           # LLM-as-a-Judge (gpt-4o-mini)
│   │   │   ├── deepeval_metrics.py    # DeepEval 메트릭 (Relevancy/Faithfulness/GEval)
│   │   │   └── tool_usage_metric.py   # 규칙 기반 Tool 사용 평가
│   │   ├── api/routes/                 # REST 엔드포인트
│   │   ├── services/
│   │   │   └── agent_service.py       # 에이전트 실행 + SSE 스트리밍
│   │   └── core/config.py             # 환경변수
│   ├── scripts/                        # 평가 스크립트
│   │   ├── run_diagnostic.py          # 규칙 기반 진단 (Day 2)
│   │   ├── run_judge_eval.py          # LLM Judge 평가 (Day 3)
│   │   ├── run_deepeval.py            # DeepEval 평가 (Day 4)
│   │   ├── generate_report.py         # 종합 리포트 생성 (Day 5)
│   │   └── eval_dataset_v2.json       # 평가 데이터셋 13건
│   └── pyproject.toml
└── ui/                                 # Frontend
    ├── src/
    │   ├── pages/ChatPage.tsx          # 채팅 화면
    │   ├── hooks/useChat.ts            # SSE 스트리밍 처리
    │   └── store/                      # Jotai 상태 관리
    └── package.json
```

### 에이전트 구성 요소

| 구성 요소 | 파일 | LangChain 개념 |
|-----------|------|----------------|
| **모델** | `real_estate_agent.py` | `ChatOpenAI(model="gpt-4.1")` |
| **도구** | `tools.py` | `@tool` 데코레이터로 정의된 3개 함수 (매매/전월세/전세가율) |
| **프롬프트** | `prompts.py` | `get_system_prompt()` 함수 — 매 요청마다 최신 날짜 주입, 기초자치단체 규칙 |
| **메모리** | `agent_service.py` | `InMemorySaver` (thread_id 기반 멀티턴) |
| **응답 포맷** | `real_estate_agent.py` | `ToolStrategy(ChatResponse)` 구조화 출력 |

### 사용 가능한 Tool

| Tool | 기능 | 데이터 소스 |
|------|------|------------|
| `search_apartment_trades` | 아파트 매매 실거래가 조회 (최대 12개월 자동 탐색, 요약 통계 포함) | data.go.kr (15126468) |
| `search_apartment_rentals` | 아파트 전월세 실거래가 조회 (최대 12개월 자동 탐색, 요약 통계 포함) | data.go.kr (15126474) |
| `calculate_jeonse_ratio` | 전세가율 계산 (매매+전세를 한번에 조회, 갭/판단근거 제공) | 내부 계산 |

---

## 5. 빠른 시작

### 사전 준비

- Python 3.11+ / Node.js 18+
- [uv](https://docs.astral.sh/uv/) / [pnpm](https://pnpm.io/)
- OpenAI API Key
- [data.go.kr](https://www.data.go.kr/) API Key (아파트매매 실거래가 상세자료 활용신청)

### Backend

```bash
cd edu_ai_agent/agent
uv sync
cp env.sample .env   # OPENAI_API_KEY, DATA_GO_KR_API_KEY 입력
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend

```bash
cd edu_ai_agent/ui
pnpm install
cp env.sample .env   # VITE_API_BASE_URL=http://localhost:8000
pnpm dev
```

### 접속

- Frontend: http://localhost:5173
- API Docs: http://localhost:8000/docs

### 환경변수

| 파일 | 변수 | 설명 |
|------|------|------|
| `agent/.env` | `OPENAI_API_KEY` | OpenAI API 키 |
| | `OPENAI_MODEL` | 사용할 모델 (기본: `gpt-4.1`) |
| | `DATA_GO_KR_API_KEY` | data.go.kr **디코딩** 키 |
| `ui/.env` | `VITE_API_BASE_URL` | Backend URL (기본: `http://localhost:8000`) |

---

## 6. 사용 예시

| 유형 | 질문 예시 | 동작 |
|------|----------|------|
| **단일 조회** | "송파구 매매가 알려줘" | `search_apartment_trades` 1회 (요약 통계 포함) |
| **전월세 조회** | "강남구 전세 시세 알려줘" | `search_apartment_rentals` 1회 |
| **전세가율** | "강남구 전세가율이 어느 정도야?" | `calculate_jeonse_ratio` 1회 (매매+전세 내부 조회) |
| **비교 분석** | "강남구랑 송파구 매매가 비교해줘" | trades 2회 호출 후 비교 |
| **시점 비교** | "용산구 작년 이맘때랑 지금 비교해줘" | trades 2회 (시점별) |
| **멀티턴** | "거기서 전세가도 알려줘" | 이전 맥락(지역)을 기억하여 조회 |
| **동 이름** | "판교 아파트 시세" | 되묻기 — "분당구 시세를 조회할까요?" |
| **광역자치단체** | "서울 아파트 매매 시세" | 되묻기 — "서울의 어느 구를 조회할까요?" |

### 데모

```
사용자: 강남구 2025년 1월 아파트 매매가 알려줘

에이전트: 2025년 1월 강남구 아파트 매매 실거래가 주요 내역입니다.
- 압구정동 미성2차 (140.9㎡, 9층): 53.0억원 (평당 약 1.24억원)
- 개포동 래미안블레스티지 (113.73㎡, 33층): 38.8억원 (평당 약 1.13억원)
- 압구정동 한양1차 (78.05㎡, 2층): 42.5억원 (평당 약 1.81억원)
...
```

---

## 7. LangChain 문서 매핑

이 프로젝트의 각 파일이 LangChain 공식 문서의 어떤 개념에 대응하는지 정리합니다.

| 프로젝트 파일 | LangChain 문서 |
|--------------|----------------|
| `real_estate_agent.py` | [Quickstart](https://docs.langchain.com/oss/python/langchain/quickstart), [Agents](https://docs.langchain.com/oss/python/langchain/agents) |
| `tools.py` | [Tools](https://docs.langchain.com/oss/python/langchain/tools) |
| `prompts.py` | [Agents > System Prompts](https://docs.langchain.com/oss/python/langchain/agents) |
| `agent_service.py` (streaming) | [Streaming](https://docs.langchain.com/oss/python/langchain/streaming/overview) |
| `agent_service.py` (memory) | [Short-term Memory](https://docs.langchain.com/oss/python/langchain/short-term-memory) |
| `ChatResponse` dataclass | [Structured Output](https://docs.langchain.com/oss/python/langchain/structured-output) |

---

## 8. 관련 문서

| 문서 | 설명 |
|------|------|
| [학습 로드맵](docs/부동산_에이전트_학습_로드맵.md) | 4주차 학습 전략 및 일자별 커리큘럼 + 2주차 진행 결과 |
| [1-2주차 학습 회고록](docs/1-2주차_학습_회고록.md) | 에이전트 구현 + 평가 파이프라인 구축 회고 + Q&A 23문항 |
| [Backend 상세](edu_ai_agent/agent/README.md) | Agent 서버 설정 및 구조 |
| [API 스펙](edu_ai_agent/agent/docs/spec.md) | API 엔드포인트 명세 |
