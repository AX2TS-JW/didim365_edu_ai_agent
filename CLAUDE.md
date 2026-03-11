# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Educational AI agent ("didim") — a chat application where users ask questions and an LLM agent responds with text, SQL code, data tables, and charts via streaming. All code lives under `edu_ai_agent/`.

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

Backend (`agent/.env`): `OPENAI_API_KEY`, `OPENAI_MODEL` (default: gpt-4.1), `CORS_ORIGINS`, `API_V1_PREFIX`
Frontend (`ui/.env`): `VITE_API_BASE_URL` (default: http://localhost:8000)

## Architecture

### Backend — FastAPI + LangChain (Python 3.11-3.13)

- **Entry**: `agent/app/main.py` — FastAPI app with CORS, logging middleware
- **Routes**: `agent/app/api/routes/` — `chat.py` (POST streaming SSE), `threads.py` (GET thread history, favorites)
- **Services**: `agent/app/services/agent_service.py` — core agent execution via LangChain `astream()`, yields JSON events with steps: `"model"` → `"tools"` → `"done"`
- **Agent**: `agent/app/agents/dummy.py` — currently an echo agent; real agent uses LangChain with OpenAI
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

Frontend sends `POST /api/v1/chat` with `{thread_id, message}`. Backend streams SSE events. Each response may include `metadata` with `sql`, `data.dataTable`, and `chart.chart_data` — frontend conditionally renders CodeEditor, GridViewer, or ChartViewer based on which metadata fields are present.

Vite dev server proxies `/api` requests to `http://localhost:8000`.
