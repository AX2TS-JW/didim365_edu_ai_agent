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
| 4 | [에이전트 설계](#4-에이전트-설계) | ReAct 메인 + StateGraph 서브에이전트 구조 |
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
- **매매 / 전월세 / 전세가율**: 질문 의도에 따라 3개 도구 중 자동 선택
- **지역명 검증**: 동/읍/면, 광역자치단체 입력 시 기초자치단체로 되묻기 (프롬프트 + 도구 2단계 방어)
- **자동 월 탐색**: 데이터 없으면 최대 12개월까지 이전 월 자동 탐색 (3개월+ 경고 표시)
- **요약 통계**: 평균가/최고가/최저가를 도구 반환에 포함
- **멀티턴 대화**: 이전 맥락을 기억하여 후속 질문 처리
- **실시간 스트리밍**: SSE를 통한 단계별 응답 (추론 → 도구 호출 → 최종 답변)
- **평가 파이프라인**: 규칙 기반 진단 → LLM Judge → DeepEval 다각도 평가 → 종합 리포트

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
                                            AGENT_MODE=react (기본)
                                                   │
                                           ┌───────▼───────┐
                                           │  LangChain     │
                                           │  ReAct Agent   │
                                           ├────────────────┤
                                           │ Tools:          │
                                           │ • 매매 실거래가  │──► data.go.kr API
                                           │ • 전월세 실거래가 │──► data.go.kr API
                                           │ • 전세가율 계산  │──► (내부 계산)
                                           │ • PDF 검색      │──► StateGraph 서브에이전트
                                           │   (search_pdf_  │     (BM25+Vector 병렬
                                           │    reports)     │      → merge → format)
                                           ├────────────────┤
                                           │ Response:       │
                                           │ ToolStrategy    │── ChatResponse 구조화 응답
                                           │ (ChatResponse)  │
                                           ├────────────────┤
                                           │ Memory:         │
                                           │ InMemorySaver   │── 대화 상태 유지
                                           └────────────────┘
```

### ReAct 에이전트 흐름 (기본, AGENT_MODE=react)

```
[사용자 질문]
      │
      ▼
 [LLM 추론]  ← 시스템 프롬프트 + 도구 목록
      │
      ├─ tool_calls 있음 → [도구 실행] → [LLM 재추론] → ... (최대 5회)
      │                      │
      │                      ├─ search_apartment_trades    ← data.go.kr API
      │                      ├─ search_apartment_rentals   ← data.go.kr API
      │                      ├─ calculate_jeonse_ratio     ← 내부 계산
      │                      └─ search_pdf_reports         ← StateGraph 서브에이전트
      │                           │
      │                           ├─ [BM25 검색] ──┐
      │                           ├─ [Vector 검색] ─┤ (병렬)
      │                           ├─ [merge] ───────┘
      │                           └─ [format] → 결과 반환
      │
      └─ ChatResponse 반환 → [SSE done 이벤트]
```

### StateGraph 에이전트 흐름 (대안, AGENT_MODE=graph)

```
                    [START]
                      │
                      ▼
                   [intent]  ← LLM: TOOL / DIRECT 판별
                      │
              ┌───────┴───────┐
              ▼               ▼
        [direct_respond]  [parse_query]  ← LLM: 질문 분석
              │               │
              ▼     ┌───────┬─┼───────────┐
            [END]   ▼       ▼ ▼           ▼
              "ambiguous" "simple" "compare" "comprehensive"
                    │       │   │           │
                    ▼       ▼   ▼           ▼
            [ask_clarification] [fetch] [fetch] → [search_pdf]
                    │       │   │           │
                    ▼       └───┴─────┬─────┘
                  [END]               ▼
                                  [respond]  ← LLM: 답변 생성
                                      │
                                      ▼
                                    [END]
```

### 통신 흐름

```
사용자 입력 → POST /api/v1/chat (thread_id + message)
         → AgentService (react, 기본) 또는 GraphAgentService (graph)
         → SSE 스트리밍: model → tools → done
         → 프론트엔드 렌더링
```

### 멀티턴 대화

```
사용자: "강남구 매매 시세"           → ReAct: search_apartment_trades 호출 → ChatResponse
사용자: "거기서 전세가도 알려줘"      → ReAct: search_apartment_rentals 호출 → ChatResponse  ← 이전 맥락 참조
사용자: "강남구 투자 전망은?"        → ReAct: search_apartment_trades + search_pdf_reports → ChatResponse
```

- ReAct 모드: `InMemorySaver` (서버 재시작 시 초기화)
- Graph 모드: `AsyncSqliteSaver` (영속 저장) + `trim_messages(max_tokens=4000)` (토큰 폭증 방지)

---

## 3. 기술 스택

| 계층 | 기술 | 역할 |
|------|------|------|
| **Frontend** | React 19, TypeScript, Vite 7, MUI 6, Jotai | 채팅 UI + 상태 관리 |
| **Backend** | FastAPI, Uvicorn, Python 3.11~3.13 | API 서버 + SSE 스트리밍 |
| **AI Agent** | LangChain 1.x, LangGraph, OpenAI GPT-4.1 | ReAct 에이전트(메인) + StateGraph 서브에이전트(PDF 검색) |
| **데이터** | data.go.kr 공공데이터 API | 아파트 매매/전월세 실거래가 |
| **캐시** | Elasticsearch | ES 캐시 (Cache-Aside 패턴, TTL) |
| **평가** | DeepEval, Opik, gpt-4o-mini (LLM Judge) | 에이전트 품질 평가 파이프라인 |
| **패키지 관리** | uv (Backend), pnpm (Frontend) | 의존성 관리 |

---

## 4. 에이전트 설계

### 프로젝트 구조

```
edu_ai_agent/
├── agent/                              # Backend
│   ├── app/
│   │   ├── agents/
│   │   │   ├── real_estate_agent.py   # ReAct 에이전트 (기본, AGENT_MODE=react)
│   │   │   │                          #   + StateGraph 서브에이전트 (search_pdf_reports 도구)
│   │   │   ├── real_estate_graph.py   # StateGraph 전체 에이전트 (대안, AGENT_MODE=graph)
│   │   │   ├── tools.py               # @tool 매매/전월세/전세가율 조회 + 지역명 검증
│   │   │   └── prompts.py            # get_system_prompt() 동적 프롬프트 + 답변 서식
│   │   ├── pipeline/                   # PDF RAG 파이프라인
│   │   │   ├── pdf_loader.py          # PyPDFLoader 페이지별 파싱
│   │   │   ├── chunker.py            # RecursiveCharacterTextSplitter (500자/100자 겹침)
│   │   │   ├── embedder.py           # OpenAI text-embedding-3-small (1536차원)
│   │   │   ├── es_client.py          # ES 벌크 적재 + 인덱스 관리
│   │   │   └── search.py             # 하이브리드 검색 (BM25 + kNN 점수 합산)
│   │   ├── evaluation/                 # 평가 체계
│   │   │   ├── llm_judge.py           # LLM-as-a-Judge (gpt-4o-mini)
│   │   │   ├── deepeval_metrics.py    # DeepEval 메트릭 (Relevancy/Faithfulness/GEval)
│   │   │   └── tool_usage_metric.py   # 규칙 기반 Tool 사용 평가
│   │   ├── api/routes/                 # REST 엔드포인트
│   │   ├── services/
│   │   │   ├── agent_service.py       # ReAct SSE 스트리밍 (기본)
│   │   │   └── graph_agent_service.py # StateGraph SSE 스트리밍 (AGENT_MODE=graph)
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

| 구성 요소 | 파일 | LangChain/LangGraph 개념 |
|-----------|------|--------------------------|
| **ReAct 에이전트** | `real_estate_agent.py` | `create_agent()` + `ToolStrategy(ChatResponse)` — LLM이 도구 선택/호출 반복 |
| **StateGraph 서브에이전트** | `real_estate_agent.py` | `SubStateGraph` — BM25+Vector 병렬 → merge → format. `@tool`로 래핑하여 `search_pdf_reports` 도구로 등록 |
| **StateGraph 전체 에이전트** | `real_estate_graph.py` | `StateGraph(AgentState)` — intent→parse→fetch→search_pdf→respond. `AGENT_MODE=graph`로 전환 가능 |
| **도구** | `tools.py` | `@tool` 데코레이터로 정의된 3개 함수 (매매/전월세/전세가율) |
| **PDF 검색** | `pipeline/search.py` | 하이브리드 검색 (BM25 + kNN 점수 합산) |
| **프롬프트** | `prompts.py` | `get_system_prompt()` 함수 — 최신 날짜 주입, 기초자치단체 규칙, 답변 서식 |
| **메모리** | `real_estate_agent.py` | `InMemorySaver` (react 기본). `real_estate_graph.py`는 `AsyncSqliteSaver` + `trim_messages` |

### 사용 가능한 Tool

| Tool | 기능 | 데이터 소스 |
|------|------|------------|
| `search_apartment_trades` | 아파트 매매 실거래가 조회 (최대 12개월 자동 탐색, 요약 통계 포함) | data.go.kr (15126468) |
| `search_apartment_rentals` | 아파트 전월세 실거래가 조회 (최대 12개월 자동 탐색, 요약 통계 포함) | data.go.kr (15126474) |
| `calculate_jeonse_ratio` | 전세가율 계산 (매매+전세를 한번에 조회, 갭/판단근거 제공) | 내부 계산 |
| `search_pdf_reports` | PDF 보고서 하이브리드 검색 (BM25 + kNN, 상위 5청크 반환) | Elasticsearch (1,082청크) |

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
사용자: 강남구 매매 시세

에이전트: 2026년 3월 강남구 아파트 매매 실거래가를 요약해 드리겠습니다.
총 30건의 거래가 있었습니다.

■ 요약 통계: 평균 34.3억원 | 최고 84.0억원 | 최저 16.2억원

- 압구정동 신현대9차 (152.28㎡): 84.0억원 (평당 약 1.82억원)
- 대치동 래미안대치팰리스 (151.31㎡): 65.0억원 (평당 약 1.42억원)
- 압구정동 미성2차 (118.63㎡): 62.5억원 (평당 약 1.74억원)
...
```

```
사용자: 판교 아파트 시세

에이전트: 판교는 동 이름으로, 실거래가 조회는 시군구 단위로 가능합니다.
판교가 속한 시군구는 성남시 분당구입니다. 분당구 아파트 시세를 조회할까요?
```

---

## 7. LangChain 문서 매핑

이 프로젝트의 각 파일이 LangChain 공식 문서의 어떤 개념에 대응하는지 정리합니다.

| 프로젝트 파일 | LangChain/LangGraph 문서 |
|--------------|--------------------------|
| `real_estate_agent.py` (ReAct, 기본) | [Agents](https://docs.langchain.com/oss/python/langchain/agents) — `create_agent()` + `ToolStrategy` |
| `real_estate_agent.py` (StateGraph 서브에이전트) | [LangGraph Quickstart](https://langchain-ai.github.io/langgraph/tutorials/introduction/) — `SubStateGraph` 병렬 fan-out/fan-in |
| `real_estate_graph.py` (StateGraph 전체, 대안) | [LangGraph Quickstart](https://langchain-ai.github.io/langgraph/tutorials/introduction/) — 조건부 엣지 |
| `real_estate_graph.py` (add_messages) | [LangGraph Messages](https://langchain-ai.github.io/langgraph/concepts/low_level/#reducers) |
| `real_estate_graph.py` (AsyncSqliteSaver) | [LangGraph Persistence](https://langchain-ai.github.io/langgraph/concepts/persistence/) |
| `tools.py` | [Tools](https://docs.langchain.com/oss/python/langchain/tools) |
| `prompts.py` | [Agents > System Prompts](https://docs.langchain.com/oss/python/langchain/agents) |
| `agent_service.py` (streaming, 기본) | [Streaming](https://docs.langchain.com/oss/python/langchain/streaming/overview) |
| `pipeline/` (PDF RAG) | [RAG Tutorial](https://docs.langchain.com/oss/python/langchain/tutorials/rag/) |

---

## 8. 관련 문서

| 문서 | 설명 |
|------|------|
| [학습 로드맵](docs/부동산_에이전트_학습_로드맵.md) | 4주차 학습 전략 및 일자별 커리큘럼 + 2~3주차 진행 결과 |
| [1-2주차 학습 회고록](docs/1-2주차_학습_회고록.md) | 에이전트 구현 + 평가 파이프라인 구축 회고 + Q&A 23문항 |
| [3주차 학습 회고록](docs/3주차_학습_회고록.md) | StateGraph 전환 + PDF RAG + 멀티턴 대화 회고 |
| [Backend 상세](edu_ai_agent/agent/README.md) | Agent 서버 설정 및 구조 |
| [API 스펙](edu_ai_agent/agent/docs/spec.md) | API 엔드포인트 명세 |
