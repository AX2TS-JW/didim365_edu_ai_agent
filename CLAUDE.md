# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

부동산 실거래가 분석 AI 에이전트 — 자연어로 부동산 질문을 하면 data.go.kr API에서 실거래가 데이터를 조회하여 분석 결과를 응답합니다. LangChain + LangGraph 기반 ReAct 패턴 에이전트. 모든 코드는 `edu_ai_agent/` 하위에 위치합니다.

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

Backend (`agent/.env`): `OPENAI_API_KEY`, `OPENAI_MODEL` (default: gpt-4.1), `DATA_GO_KR_API_KEY` (data.go.kr 디코딩키), `CORS_ORIGINS`, `API_V1_PREFIX`
Frontend (`ui/.env`): `VITE_API_BASE_URL` (default: http://localhost:8000)

## Architecture

### Backend — FastAPI + LangChain (Python 3.11-3.13)

- **Entry**: `agent/app/main.py` — FastAPI app with CORS, logging middleware
- **Routes**: `agent/app/api/routes/` — `chat.py` (POST streaming SSE), `threads.py` (GET thread history, favorites)
- **Services**: `agent/app/services/agent_service.py` — core agent execution via LangChain `astream()`, yields JSON events with steps: `"model"` → `"tools"` → `"done"`
- **Agent**: `agent/app/agents/real_estate_agent.py` — `create_agent()` + `ToolStrategy(ChatResponse)` + `InMemorySaver`
- **Tools**: `agent/app/agents/tools.py` — `@tool` 데코레이터로 매매/전월세/전세가율 조회 (data.go.kr). 자동 월 탐색(최대 12개월), 요약 통계(평균/최고/최저), 지역명 검증(동/읍/면/시/도 감지) 포함
- **Prompts**: `agent/app/agents/prompts.py` — `get_system_prompt()` 함수로 매 요청마다 최신 날짜 주입. 기초자치단체 단위 조회 규칙, Tool 호출 최대 5회 제한
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

- `ChatResponse` dataclass 클래스명은 반드시 "ChatResponse"여야 함 — `agent_service.py`의 tool_calls 파싱과 일치 필요
- data.go.kr API는 **디코딩** 키 사용 (인코딩 키 사용 시 이중 인코딩 문제 발생)
- API 응답 필드는 영문: `aptNm`, `dealAmount`, `excluUseAr`, `floor`, `umdNm`, `dealDay`
- `cdealType`이 존재하면 해제(취소)된 거래 → 필터링 대상
- 시스템 프롬프트는 `get_system_prompt()` 함수로 매 요청마다 최신 날짜 주입 (모듈 레벨 변수 아님)
- Tool 호출 최대 5회 제한 (프롬프트 + `MAX_TOOL_CALLS` 코드 이중 방어)
- 실거래가 API는 **기초자치단체(구/시/군)** 단위로만 조회 가능. 동/읍/면, 광역자치단체는 도구에서 감지 후 안내
- 도구는 데이터 없으면 최대 12개월까지 자동 탐색, 3개월 이상 전 데이터는 "참고용" 경고 표시
- 도구 반환에 요약 통계(평균가/최고가/최저가/총 건수) 포함
