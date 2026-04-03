# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

부동산 실거래가 분석 AI 에이전트 — 자연어로 부동산 질문을 하면 data.go.kr API에서 실거래가 데이터를 조회하고, PDF 보고서에서 전문가 분석을 검색하여 종합 답변합니다. Deep Agent SDK(메인, 기본) + LangGraph StateGraph 서브에이전트(PDF 하이브리드 검색) 구조. 모든 코드는 `edu_ai_agent/` 하위에 위치합니다.

## Commands

### Backend (agent)

```bash
cd edu_ai_agent/agent
uv sync                          # Install/update Python dependencies
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload   # Run dev server
```

API docs available at `http://localhost:8000/docs`.

### Frontend (ui)

```bash
cd edu_ai_agent/ui
pnpm install                     # Install dependencies
pnpm dev                         # Dev server at http://localhost:5173
pnpm build                       # Production build
pnpm lint                        # ESLint
```

### Environment Variables

Backend (`agent/.env`): `OPENAI_API_KEY`, `OPENAI_MODEL` (default: gpt-4.1-mini), `DATA_GO_KR_API_KEY` (data.go.kr 디코딩키), `CORS_ORIGINS`, `API_V1_PREFIX`, `AGENT_MODE` (default: deep, 옵션: deep/react/graph)
Frontend (`ui/.env`): `VITE_API_BASE_URL` (default: http://localhost:8000)

## Architecture

### Backend — FastAPI + LangChain (Python 3.11-3.13)

- **Entry**: `agent/app/main.py` — FastAPI app with CORS, logging middleware
- **Routes**: `agent/app/api/routes/` — `chat.py` (POST streaming SSE), `threads.py` (GET thread history, favorites)
- **Services**: `agent/app/services/agent_service.py` — Deep Agent/ReAct 에이전트 스트리밍을 SSE 형식(model→tools→done)으로 변환 (기본). MAX_TOOL_CALLS=50(deep)/10(react). ChatResponse yield 후 stream을 background drain하여 GeneratorExit 방지. 체크포인트 정합성 패치(미완료 tool_calls → 더미 ToolMessage). `agent/app/services/graph_agent_service.py` — StateGraph 전체 에이전트용 (`AGENT_MODE=graph`일 때)
- **Agent (Deep Agent, 기본)**: `agent/app/agents/real_estate_agent.py` — `create_deep_agent()` + 도구 4개(직접 호출) + 서브에이전트 3개(data-collector, analyst, reporter) + 장기 기억(AGENTS.md). `AGENT_MODE=deep`(기본)
- **Agent (ReAct, 대안)**: `agent/app/agents/real_estate_agent.py` — `create_agent()` + `ToolStrategy(ChatResponse)`. `AGENT_MODE=react`
- **Agent (StateGraph, 대안)**: `agent/app/agents/real_estate_graph.py` — 전체 흐름을 StateGraph로 제어. 노드: intent→parse→fetch→search_pdf→respond. `AGENT_MODE=graph`
- **Tools**: `agent/app/agents/tools.py` — `@tool` 데코레이터로 매매/전월세/전세가율 조회 (data.go.kr). 자동 월 탐색(최대 12개월), 요약 통계(평균/최고/최저), 지역명 검증(동/읍/면/시/도 감지) 포함. `real_estate_agent.py`에 정의된 `search_pdf_reports`는 StateGraph 서브에이전트를 `@tool`로 래핑한 PDF 검색 도구
- **Pipeline (PDF RAG)**: `agent/app/pipeline/` — PDF 파싱(`pdf_loader.py`), 청킹(`chunker.py`), 임베딩(`embedder.py`), ES 적재(`es_client.py`), 하이브리드 검색(`search.py`) — BM25 + kNN 점수 합산
- **Prompts**: `agent/app/agents/prompts.py` — `get_system_prompt()` 함수로 매 요청마다 최신 날짜 주입. 기초자치단체 단위 조회 규칙, 서브에이전트 위임 기준(3개 이상 지역 비교 시 task()), 답변 서식 가이드
- **AGENTS.md**: `agent/app/agents/AGENTS.md` — Deep Agent 장기 기억 파일. 전세가율 판단 기준, 투자 판단 요소, 서울 지역 특성. MemoryMiddleware가 자동 로드 (정적 read-only)
- **Evaluation**: `agent/app/evaluation/` — LLM Judge (`llm_judge.py`), DeepEval 메트릭 (`deepeval_metrics.py`), Tool 사용 메트릭 (`tool_usage_metric.py`)
- **Scripts**: `agent/scripts/` — 진단(`run_diagnostic.py`), Judge 평가(`run_judge_eval.py`), DeepEval 평가(`run_deepeval.py`), 종합 리포트(`generate_report.py`)
- **Config**: `agent/app/core/config.py` — Pydantic Settings loaded from `.env`
- **Mock data**: `agent/app/data/` — JSON files for threads and favorite questions

### Frontend — React 19 + TypeScript + Vite

- **Entry**: `ui/src/main.tsx` → `App.tsx` (router)
- **Pages**: `InitPage` (landing), `ChatPage` (main chat with code/data/chart panels)
- **State**: Jotai atoms in `ui/src/store/` — `messageAtom`, `answerAtom`, `questionAtom`, `threadById`
- **Hooks**: `useChat` (chat logic + streaming), `useHistory` (conversation history)
- **API client**: `ui/src/services/common.ts` — Axios with SSE streaming via `fetchEventSource`, auto snake_case→camelCase conversion
- **Routes**: `/` (InitPage), `/chat` (ChatPage), `/dashboard`, `/setting`
- **i18n**: Korean (`ko.json`) and English (`en.json`) in `ui/src/constants/i18n/`
- **UI libs**: MUI 6, CodeMirror (SQL display), Highcharts (charts)
- **Path alias**: `@` → `./src` (configured in vite.config.ts)

### Communication Flow

Frontend sends `POST /api/v1/chat` with `{thread_id, message}`. Backend streams SSE events with steps: `model` → `tools` → `done`. Agent uses `astream(stream_mode="updates")` for 실시간 스트리밍. 응답의 `metadata`에 따라 프론트엔드가 조건부 렌더링 (코드/테이블/차트).

Vite dev server proxies `/api` requests to `http://localhost:8000`.

### Key Implementation Notes

- **Deep Agent 메인 (기본)**: `create_deep_agent()`로 생성. 도구 4개 직접 호출(단순 질문) + 서브에이전트 3개(data-collector/analyst/reporter, 복합 질문 task() 위임) + AGENTS.md 장기 기억. SDK 내장 미들웨어(TodoListMiddleware, MemoryMiddleware, PatchToolCallsMiddleware) 자동 개입
- **AGENT_MODE 환경변수**: `deep` (기본, Deep Agent) / `react` (ReAct) / `graph` (전체 StateGraph). `chat.py`에서 분기. **기본값은 반드시 한 곳에서 관리** (chat.py와 agent_service.py 모두 "deep")
- **recursion_limit과 MAX_TOOL_CALLS 관계**: Deep Agent는 도구 1회당 3~4 recursion 스텝 소모 (미들웨어 개입). `DEEPAGENT_RECURSION_LIMIT=200`, `MAX_TOOL_CALLS=50`(deep)/`10`(react). 공식: `recursion_limit ≥ MAX_TOOL_CALLS × 4`
- **SSE stream drain**: ChatResponse yield 후 `asyncio.ensure_future()`로 남은 LangGraph stream을 background에서 drain. GeneratorExit/CancelledError 방지
- **체크포인트 정합성 패치**: MAX_TOOL_CALLS 강제 종료 시 미완료 tool_calls를 더미 ToolMessage로 패치하여 다음 요청의 400 에러 방지
- **graph 모드 영속 체크포인트**: `AsyncSqliteSaver`로 대화 히스토리 영속 저장. deep/react 모드는 `InMemorySaver`
- `ChatResponse` dataclass 클래스명은 반드시 "ChatResponse"여야 함 — `agent_service.py`의 tool_calls 파싱과 일치 필요
- data.go.kr API는 **디코딩** 키 사용 (인코딩 키 사용 시 이중 인코딩 문제 발생)
- API 응답 필드는 영문: `aptNm`, `dealAmount`, `excluUseAr`, `floor`, `umdNm`, `dealDay`
- `cdealType`이 존재하면 해제(취소)된 거래 → 필터링 대상
- 시스템 프롬프트는 `get_system_prompt()` 함수로 매 요청마다 최신 날짜 주입 (모듈 레벨 변수 아님)
- `search_pdf_reports` 도구: `real_estate_agent.py`에 정의. StateGraph 서브에이전트(BM25+Vector 병렬→merge→rerank→format)를 `@tool`로 래핑. `pipeline/search.py`의 `search_bm25`, `search_vector` 사용
- 실거래가 API는 **기초자치단체(구/시/군)** 단위로만 조회 가능. 동/읍/면, 광역자치단체는 도구에서 감지 후 안내
- 도구는 데이터 없으면 최대 12개월까지 자동 탐색, 3개월 이상 전 데이터는 "참고용" 경고 표시
- 도구 반환에 요약 통계(평균가/최고가/최저가/총 건수) 포함
- SSE 스트리밍: `fetchEventSource`에 `controller.abort()` 적용 (done/close/error 시). `openWhenHidden: true`로 탭 전환 시 재연결 방지
- 프롬프트 인젝션 3계층 방어: `detect_injection()` 입력 필터링 → 시스템 프롬프트 보안 규칙 → `check_leakage()` 출력 필터링
