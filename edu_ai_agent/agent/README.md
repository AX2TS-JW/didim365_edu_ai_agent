# 부동산 실거래가 AI 에이전트

자연어로 부동산 질문을 하면 data.go.kr API에서 실거래가 데이터를 조회하고, PDF 보고서에서 전문가 분석을 검색하여 종합 답변하는 AI 에이전트입니다.

## 기술 스택

- **Backend**: FastAPI + LangChain + LangGraph + Deep Agent SDK
- **Frontend**: React 19 + TypeScript + Vite + MUI 6
- **LLM**: OpenAI (gpt-4.1-mini)
- **검색**: Elasticsearch (BM25 + kNN 하이브리드)
- **트레이싱**: Opik
- **패키지 관리**: uv (Python), pnpm (Node.js)

## 아키텍처

```
create_deep_agent() — 메인 (Supervisor)
├── 도구 4개 (단순 질문 직접 호출)
│   ├── search_apartment_trades      매매 실거래가 조회
│   ├── search_apartment_rentals     전월세 실거래가 조회
│   ├── calculate_jeonse_ratio       전세가율 계산
│   └── search_pdf_reports           PDF 하이브리드 검색 (StateGraph 서브에이전트)
├── SDK 내장 도구 (자동)
│   ├── write_todos / read_todos     작업 분해 + 진행 추적
│   ├── write_file / read_file       VFS (가상 파일 시스템)
│   └── task()                       서브에이전트 위임
├── 서브에이전트 3개 (복합 질문 위임)
│   ├── data-collector               데이터 수집 + VFS 저장
│   ├── analyst                      비교 분석
│   └── reporter                     종합 리포트 작성
├── 장기 기억 (AGENTS.md)
│   ├── 전세가율 판단 기준
│   ├── 투자 판단 요소
│   └── 서울 지역 특성
└── 체크포인터 (InMemorySaver)
```

### 에이전트 모드 (AGENT_MODE)

| 모드 | 설명 | 주차 |
|------|------|------|
| `deep` (기본) | Deep Agent SDK + 서브에이전트 + VFS + 장기 기억 | 4주차 |
| `react` | ReAct 패턴 + StateGraph PDF 서브에이전트 | 1~3주차 |
| `graph` | 전체 StateGraph 제어 (intent→parse→fetch→respond) | 3주차 |

### 통신 흐름

```
Frontend → POST /api/v1/chat {thread_id, message}
         ← SSE stream: model → tools → done
```

## 환경 준비

### 1. 사전 요구사항

- Python 3.11~3.13
- Node.js 18+
- uv 패키지 매니저
- pnpm

### 2. Backend 설치 및 실행

```bash
cd edu_ai_agent/agent
uv sync
cp env.sample .env  # .env 파일 생성 후 API 키 입력
uv run uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 3. Frontend 설치 및 실행

```bash
cd edu_ai_agent/ui
pnpm install
pnpm dev  # http://localhost:5173
```

### 4. 환경 변수 (.env)

```env
# 필수
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4.1-mini
API_V1_PREFIX=/api/v1
DATA_GO_KR_API_KEY=your_decoded_key_here  # 디코딩 키 사용

# Elasticsearch (PDF 검색용)
ES_URL=https://your-es-url
ES_USER=elastic
ES_PASSWORD=your_password

# Opik 트레이싱 (선택)
OPIK__URL_OVERRIDE=https://your-opik-url/api
OPIK__PROJECT=your-project-name
```

## 프로젝트 구조

```
edu_ai_agent/
├── agent/                          # Backend
│   ├── app/
│   │   ├── api/routes/             # chat.py (SSE), threads.py
│   │   ├── agents/
│   │   │   ├── real_estate_agent.py   # Deep Agent + ReAct + 서브에이전트 정의
│   │   │   ├── real_estate_graph.py   # StateGraph 에이전트
│   │   │   ├── tools.py               # @tool 데코레이터 (매매/전세/전세가율)
│   │   │   ├── prompts.py             # 시스템 프롬프트
│   │   │   └── AGENTS.md              # 장기 기억 (도메인 지식)
│   │   ├── services/
│   │   │   ├── agent_service.py       # Deep/ReAct SSE 스트리밍
│   │   │   └── graph_agent_service.py # StateGraph SSE 스트리밍
│   │   ├── pipeline/                  # PDF RAG 파이프라인
│   │   ├── evaluation/                # LLM Judge, DeepEval
│   │   ├── core/config.py             # Pydantic Settings
│   │   └── main.py                    # FastAPI 진입점
│   └── scripts/                       # 평가/진단 스크립트
├── ui/                             # Frontend (React + Vite)
└── docs/                           # 회고록, 이슈 분석
```

## API 엔드포인트

| Method | Path | 설명 |
|--------|------|------|
| POST | `/api/v1/chat` | 자연어 질문 → SSE 스트리밍 답변 |
| GET | `/api/v1/threads` | 대화 히스토리 목록 |
| GET | `/api/v1/threads/favorites` | 추천 질문 목록 |
| GET | `/health` | 헬스 체크 |

## 주차별 학습 진행

| 주차 | 주제 | 핵심 기술 |
|------|------|----------|
| 1주차 | AI 에이전트 기초 | ReAct 패턴, @tool, LangChain create_agent() |
| 2주차 | 평가 파이프라인 | Opik 트레이싱, LLM Judge, DeepEval |
| 3주차 | 고급 에이전트 | StateGraph, PDF RAG, 멀티턴, 하이브리드 검색 |
| 4주차 | Deep Agent | create_deep_agent(), 서브에이전트, VFS, 장기 기억 |
