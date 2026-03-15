# LangChain 문서 학습 가이드

> 프로젝트 코드 기준으로 LangChain 공식 문서를 이해하기 위한 가이드입니다.
> 문서 URL: https://docs.langchain.com/oss/python/langchain/

---

## 1. 문서 사이트 구조

LangChain 문서 사이트는 **3개의 프로젝트**가 합쳐져 있습니다.

```
docs.langchain.com/
├── langchain/          ← 지금 쓰고 있는 것 (1~2주차)
│   ├── quickstart      "5분 만에 에이전트 만들기"
│   ├── agents          에이전트 = 뇌 (판단하고 도구를 고르는 놈)
│   ├── tools           도구 = 손 (API 호출, 검색 등 실제 행동)
│   ├── structured-output  응답 형태를 정해주는 것
│   ├── streaming       실시간 스트리밍
│   └── short-term-memory  대화 기억
│
├── langgraph/          ← 3주차에 배울 것
│   └── (복잡한 흐름 제어 — "A 다음에 B 하고, 조건에 따라 C로 분기")
│
└── deepagents/         ← 4주차에 배울 것
    └── (에이전트가 에이전트를 부리는 구조)
```

**지금은 `langchain/` 안의 문서만 보면 됩니다.**

---

## 2. 핵심 용어 정리

프로젝트 코드에 직접 대입한 설명입니다.

```
사용자: "강남구 매매가 알려줘"
          │
          ▼
     ┌─────────┐
     │  Agent   │  = 사람으로 치면 "머리"
     │          │  → 질문을 이해하고, 어떤 도구를 쓸지 판단
     │          │  → real_estate_agent.py
     └────┬────┘
          │ "매매니까 매매 도구를 쓰자"
          ▼
     ┌─────────┐
     │  Tool    │  = 사람으로 치면 "손"
     │          │  → 실제로 API를 호출하는 함수
     │          │  → tools.py의 search_apartment_trades()
     └────┬────┘
          │ API 결과 반환
          ▼
     ┌─────────┐
     │ Response │  = 결과물
     │ Format   │  → 답변을 정해진 형태로 포장
     │          │  → ChatResponse 데이터클래스
     └─────────┘
```

| 용어 | 한마디 | 프로젝트에서 |
|------|--------|-------------|
| **Agent** | 판단하는 뇌 | `create_agent()` |
| **Tool** | 실행하는 손 | `@tool` 함수들 |
| **System Prompt** | 뇌에 주는 업무 지침서 | `prompts.py` |
| **Memory** | 이전 대화 기억 | `InMemorySaver` |
| **Structured Output** | 답변 포맷 규칙 | `ToolStrategy(ChatResponse)` |
| **Streaming** | 실시간 중계 | `astream()` |
| **Chain** | 단순 직선 파이프라인 (A→B→C) | 지금은 안 씀 |
| **Graph** | 조건 분기가 있는 파이프라인 | 3주차에 배움 |

**Chain vs Agent 차이**:
- Chain: "무조건 A→B→C 순서로 실행해" (고정)
- Agent: "상황 보고 네가 알아서 판단해" (유동) ← 지금 이거

---

## 3. 문서 전체 지도 — 주차별 학습 범위

```
docs.langchain.com/oss/python/langchain/
│
├── 🟢 지금 봐야 할 것 (1주차에 해당)
│   ├── quickstart          전체 흐름 한눈에 보기
│   ├── agents              에이전트 만드는 법
│   ├── tools               도구 만드는 법
│   ├── structured-output   답변 형태 지정
│   ├── streaming           실시간 응답
│   └── short-term-memory   대화 기억
│
├── 🟡 곧 볼 것 (2~3주차)
│   ├── long-term-memory    장기 기억
│   ├── rag                 검색 보강 생성 (ES + PDF)
│   ├── retrieval           검색 기법
│   ├── middleware           중간 처리 로직
│   └── test/evals          평가/채점
│
└── 🔴 나중에 볼 것 (4주차 이후)
    ├── multi-agent          에이전트 여러 개 협업
    ├── human-in-the-loop    사람 승인 끼우기
    ├── guardrails           안전장치
    └── mcp                  외부 도구 연결 프로토콜
```

---

## 4. Quickstart 문서 ↔ 프로젝트 코드 매핑

> 문서: https://docs.langchain.com/oss/python/langchain/quickstart

Quickstart의 7단계가 프로젝트 코드의 어디에 해당하는지:

| Step | 문서가 말하는 것 | 프로젝트 코드 | 파일 |
|------|----------------|-------------|------|
| 1 | Install | `uv sync` | `pyproject.toml` |
| 2 | System Prompt | 에이전트에게 주는 지침서 | `prompts.py` |
| 3 | Tools | `@tool` 함수 만들기 | `tools.py` |
| 4 | Model | LLM 모델 선택 | `ChatOpenAI(model="gpt-4.1")` |
| 5 | Response Format | 답변 형태 지정 | `ToolStrategy(ChatResponse)` |
| 6 | Memory | 대화 기억 장치 | `InMemorySaver()` |
| 7 | Assembly | 위 재료를 조립 | `create_agent(model, tools, ...)` |

`real_estate_agent.py`를 위에서 아래로 읽으면 이 순서와 1:1로 매칭됩니다:

```python
# Step 4: 모델
model = ChatOpenAI(model="gpt-4.1")

# Step 5: 응답 포맷
@dataclass
class ChatResponse:        # ← 이 이름이 "ChatResponse"여야 함
    message_id: str
    content: str
    metadata: dict

# Step 6: 메모리
checkpointer = InMemorySaver()

# Step 7: 조립 — 위 재료를 다 합침
agent = create_agent(
    model=model,                              # 뇌의 엔진
    tools=[search_apartment_trades, ...],     # 쓸 수 있는 도구들
    system_prompt=system_prompt,              # 업무 지침서
    response_format=ToolStrategy(ChatResponse), # 답변 규칙
    checkpointer=checkpointer,               # 기억 장치
)
```

---

## 5. 카테고리별 세부 항목 + 프로젝트 코드 매핑

### 5-1. Agents (에이전트)

> 문서: https://docs.langchain.com/oss/python/langchain/agents

| 문서 소제목 | 쉬운 설명 | 프로젝트에서 |
|------------|----------|-------------|
| **Core components** | | |
| ├ Model (Static) | LLM 모델을 고정으로 설정 | `ChatOpenAI(model="gpt-4.1")` ✅ 지금 이거 |
| ├ Model (Dynamic) | 상황에 따라 모델을 바꿈 | 안 씀 |
| ├ Tools (Static) | 도구 목록을 고정 | `tools=[search_trades, search_rentals]` ✅ |
| ├ Tools (Dynamic) | 상황에 따라 도구를 추가/제거 | 안 씀 |
| ├ System Prompt | 지침서 | `system_prompt` ✅ |
| ├ Structured Output | 답변 형태 | `ToolStrategy` vs `ProviderStrategy` |
| └ Memory | 기억 장치 | `InMemorySaver` ✅ |
| **Advanced** | | |
| ├ ReAct Loop | 추론→행동→관찰 반복 | 자동으로 돌아감 (설정 불필요) |
| ├ Error Handling | Tool 실행 실패 시 처리 | 기본값 사용 중 |
| └ Middleware | 실행 전후에 끼워넣는 로직 | 안 씀 (2주차 Opik에서 도입) |

> **팁**: `Static`이 붙은 항목만 먼저 읽기. `Dynamic`은 고급 기능.

### 5-2. Tools (도구)

> 문서: https://docs.langchain.com/oss/python/langchain/tools

| 문서 소제목 | 쉬운 설명 | 프로젝트에서 |
|------------|----------|-------------|
| **Create tools** | | |
| ├ Basic definition | `@tool` + 함수 + docstring | `tools.py` ✅ 지금 이거 |
| ├ Customize name | 도구 이름 바꾸기 | 안 씀 (함수명이 곧 이름) |
| ├ Customize description | 도구 설명 바꾸기 | docstring이 자동으로 설명이 됨 ✅ |
| └ Advanced schema | 파라미터 타입을 세밀하게 지정 | 안 씀 |
| **Access context** | | |
| ├ Short-term memory | Tool 안에서 이전 대화 참조 | 안 씀 |
| ├ Long-term memory | Tool 안에서 장기 기억 참조 | 안 씀 (4주차) |
| └ Stream writer | Tool에서 실시간 메시지 보내기 | 안 씀 |
| **Return values** | | |
| ├ String | 문자열 반환 | `tools.py` ✅ 지금 이거 |
| ├ Object | 구조화된 객체 반환 | 안 씀 |
| └ Command | 상태를 직접 조작 | 안 씀 (고급) |

> **팁**: `tools.py`를 열어놓고 문서의 "Basic definition" 예제와 비교하기:
>
> ```python
> # 문서 예제                    # 프로젝트 코드
> @tool                         @tool
> def get_weather(city):        def search_apartment_trades(region, year_month):
>     """날씨 조회"""               """매매 실거래가 조회"""
>     return f"{city}: 맑음"       return header + summaries
> ```

### 5-3. Structured Output (구조화된 출력)

> 문서: https://docs.langchain.com/oss/python/langchain/structured-output

| 문서 소제목 | 쉬운 설명 | 프로젝트에서 |
|------------|----------|-------------|
| ToolStrategy | Tool 호출 방식으로 형태를 강제 | `ToolStrategy(ChatResponse)` ✅ 지금 이거 |
| ProviderStrategy | LLM 제공사의 자체 기능 사용 | 안 씀 |
| Error handling | 형태가 안 맞을 때 처리 | 기본값 사용 |

> **핵심**: `ChatResponse` 클래스의 필드 3개(message_id, content, metadata)가 곧 LLM이 반드시 채워야 하는 답변 형태. `agent_service.py`에서 이 이름으로 파싱하므로 클래스명을 바꾸면 안 됨.

### 5-4. Streaming (스트리밍)

> 문서: https://docs.langchain.com/oss/python/langchain/streaming/overview

| 문서 소제목 | 쉬운 설명 | 프로젝트에서 |
|------------|----------|-------------|
| Agent progress | 에이전트 진행 단계별 이벤트 | `astream(stream_mode="updates")` ✅ |
| LLM tokens | 글자 하나하나 실시간 전송 | SSE로 프론트에 전달 ✅ |
| Custom updates | 커스텀 이벤트 보내기 | 안 씀 |
| Stream from sub-agents | 하위 에이전트의 스트림 | 안 씀 (4주차) |

> **핵심**: `agent_service.py`의 `process_query()` 메서드에서 `async for event in agent.astream(...)` 부분. 이벤트가 오면 `model` → `tools` → `done` 단계로 나눠서 SSE로 전송.

### 5-5. Short-term Memory (단기 기억)

> 문서: https://docs.langchain.com/oss/python/langchain/short-term-memory

| 문서 소제목 | 쉬운 설명 | 프로젝트에서 |
|------------|----------|-------------|
| Overview | thread_id로 대화를 구분 | `config={"configurable": {"thread_id": ...}}` ✅ |
| In production | 운영환경에서 저장소 선택 | `InMemorySaver` (서버 재시작하면 사라짐) ✅ |
| Trim messages | 오래된 대화 잘라내기 | 안 씀 |
| Summarize messages | 이전 대화를 요약해서 압축 | 안 씀 |
| Delete messages | 특정 대화 삭제 | 안 씀 |

> **핵심**: "강남구 매매가 알려줘" → "거기서 전세가도" 이 멀티턴이 되는 이유가 `InMemorySaver`가 thread_id별로 대화를 기억하고 있기 때문. 3주차에 `SqliteSaver`로 바꾸면 서버 재시작해도 유지.

---

## 6. 추천 학습 순서

```
Step 1: quickstart 한번 쭉 읽기 (전체 감 잡기)
   ↓
Step 2: tools → 프로젝트 tools.py와 비교
   ↓
Step 3: agents → 프로젝트 real_estate_agent.py와 비교
   ↓
Step 4: structured-output → ChatResponse 이해
   ↓
Step 5: short-term-memory → 멀티턴 동작 원리
   ↓
Step 6: streaming → agent_service.py의 astream() 이해
```

각 단계에서 **문서 예제 코드 → 프로젝트 코드**를 나란히 놓고 보면 연결됩니다.

---

## 7. RAG와 청킹 (3주차 사전 지식)

### RAG(Retrieval-Augmented Generation)란?

LLM 모델 자체를 바꾸는 게 아니라, 질문이 들어올 때 관련 내용을 **검색(Retrieval)**해서 "이거 참고해서 답변해"라고 LLM에 같이 넘겨주는 방식.

### 청킹(Chunking)이란?

PDF 같은 긴 문서를 검색 가능한 단위로 잘라서 저장하는 과정. 잘라는 크기가 검색 품질을 결정.

- **너무 작게 자르면**: 맥락이 없어서 검색 결과가 무의미
- **너무 크게 자르면**: 관련 없는 내용이 딸려오고 토큰 낭비
- **적절하게 자르면**: 맥락 보존 + 검색 정밀도 확보

### 청킹 근거 자료

| 출처 | 핵심 내용 | URL |
|------|----------|-----|
| arXiv 논문 (2505.21700) | 청크 크기의 영향이 임베딩 모델 선택만큼 크다 | https://arxiv.org/html/2505.21700v2 |
| AI21 Labs | 질문에 맞는 청크 크기 시 검색 정확도 20~40% 향상 | https://www.ai21.com/blog/query-dependent-chunking/ |
| Elasticsearch 공식 | "청크는 작게 하되 유용한 맥락은 보존하라" | https://www.elastic.co/search-labs/blog/chunking-strategies-elasticsearch |
| LangChain 공식 | RecursiveCharacterTextSplitter를 기본 권장 | https://docs.langchain.com/oss/python/integrations/splitters |

### 3주차 RAG 파이프라인 전체 흐름

```
1. PDF 로드        → PyPDFLoader로 PDF 텍스트 추출
2. 청킹           → 텍스트를 의미 단위로 잘라서 조각냄
3. 임베딩          → 각 조각을 벡터(숫자 배열)로 변환
4. ES에 저장       → 벡터 + 원문 텍스트를 ES에 인덱싱
5. 검색 (질문 시)   → 질문과 가장 유사한 조각을 찾아옴
6. LLM에 전달      → 찾아온 조각을 LLM에 넘겨서 답변 생성
```

---

## 부록: Elasticsearch 도입 이유 (1주차 vs 3주차)

| 주차 | ES 역할 | 비유 |
|------|---------|------|
| 1주차 | 실거래가 **캐시 저장소** | 자주 보는 책을 내 책상에 갖다놓는 것 |
| 3주차 | **하이브리드 검색 엔진** | 수치 데이터 + PDF 텍스트를 동시에 검색 |

현재 구현(API 직접 호출)은 매번 도서관에 가서 책을 찾아오는 방식.
ES 도입 후에는 미리 책상에 갖다놓고 바로 펼쳐보는 방식.
