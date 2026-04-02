# 3주차 — MCP 토큰 비용 비교 테스트 & 코드 검증

> 테스트 일자: 2026-04-02
> 테스트 스크립트: `agent/scripts/mcp_token_compare.py`

---

## 1. MCP 토큰 비용 비교

### 테스트 설계

같은 질문("LangGraph StateGraph의 add_node 파라미터를 설명해줘")을 3가지 조건으로 OpenAI API 호출:

| 시나리오 | 조건 |
|---|---|
| 테스트 1 | 도구 없이 LLM 기존 지식만 |
| 테스트 2 | MCP 도구 스키마만 등록 (호출하지 않음) |
| 테스트 3 | MCP 검색 결과를 컨텍스트에 포함 |

### 결과 (gpt-4.1-mini)

| 시나리오 | 입력 토큰 | 출력 토큰 | 총 토큰 | 입력 증가 |
|---|---|---|---|---|
| MCP 미사용 (기존 지식) | 43 | 386 | 429 | 기준 |
| MCP 도구 등록만 (스키마) | 195 | 23 | 218 | +152 (+353%) |
| MCP 검색 결과 포함 (풀) | 357 | 375 | 732 | +314 (+730%) |

### 핵심 발견

1. **도구 스키마만 연결해도 입력 토큰 3.5배** — MCP 서버를 연결만 해도 매 요청마다 +152 토큰. 필요할 때만 연결하는 것이 비용 최적화.

2. **검색 결과까지 포함하면 7.3배** — 실제 도구 호출 결과가 컨텍스트에 들어가면 +314 토큰. `get_page`로 전체 문서 가져오면 수천~수만 토큰 가능.

3. **테스트 2에서 LLM이 직접 답하지 않고 tool_call 반환** — 도구 스키마를 주니 LLM이 "도구로 검색하자"고 판단. 도구 등록이 LLM 행동 자체를 바꿈.

### 답변 품질 비교

| | MCP 미사용 | MCP 사용 |
|---|---|---|
| 정확도 | **hallucination 발생** — `node_id`, `node_type`, `data`, `metadata` 등 존재하지 않는 파라미터 날조 | 공식 문서 기반 정확한 답변 |
| 비용 | 429 토큰 | 732 토큰 (+76%) |

**결론**: 토큰 76% 더 들지만, MCP 없이는 아예 틀린 답을 자신 있게 제공. LLM이 잘 아는 영역은 도구 없이, 모르거나 최신 정보 필요한 영역만 MCP 연결하는 것이 비용 최적화의 핵심.

---

## 2. MCP 동작 원리 정리

```
1. MCP 서버 → 내 로컬:    "나한테 이런 도구들 있어" (tools/list → 명세 전달)
2. 내 로컬 → LLM:         "사용자 질문 + 도구 명세" (매 요청마다 함께 전달)
3. LLM → 내 로컬:         "search_docs를 query='add_node'로 호출해" (LLM이 판단)
4. 내 로컬 → MCP 서버:    "search_docs 실행해줘" (실제 실행 요청)
5. MCP 서버 → 내 로컬:    "결과 여기 있어" (실행 결과)
6. 내 로컬 → LLM:         "결과 이거야, 이걸로 답변해" (결과 전달)
```

토큰 비용 발생 지점:
- 2번: 도구 명세 자체 (+152 토큰, 매 요청마다)
- 3번: LLM 판단 결과 (소량)
- 6번: 실행 결과 전달 (+314~수만 토큰)

LangChain `@tool` + `bind_tools()`와 완전히 같은 구조. MCP는 "도구가 외부 서버에 있다"는 점만 다름.

---

## 3. 코드 검증 — MCP 문서 대조

LangChain Docs MCP(`docs-langchain`)로 공식 문서와 구현체를 대조한 결과:

### 문제: `real_estate_graph.py`의 messages reducer 누락

```python
# 현재 (문제)
class AgentState(TypedDict):
    messages: list           # reducer 없음 → 덮어쓰기

# 공식 문서 권장
from typing import Annotated
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]  # 누적
```

**왜 문제인가**: reducer 없으면 노드가 `{"messages": [새메시지]}` 반환 시 기존 대화 히스토리가 통째로 덮어쓰여짐. 현재는 어떤 노드도 messages를 반환하지 않아서 동작에 문제 없지만, 3주차 목요일 Persistence(멀티턴) 구현 시 반드시 수정 필요.

**참고**: `real_estate_agent.py`의 `create_agent()`는 내부적으로 add_messages reducer를 자동 세팅하므로 문제 없음.

### SqliteSaver와 reducer는 별개

| | reducer (add_messages) | checkpointer (SqliteSaver) |
|---|---|---|
| 역할 | 새 메시지를 기존에 어떻게 합칠지 | 합친 결과를 어디에 저장할지 |
| 없으면 | 기존 대화 덮어씀 | 서버 재시작 시 대화 유실 |

SqliteSaver가 있어도 reducer 없으면 망가진 State를 그대로 저장함.

### 멀티턴 메시지 트리밍 (서킷브레이커) 필요

add_messages는 무한 누적 → MAX_TOOL_CALLS=5처럼 메시지에도 제한 필요:

```python
from langchain_core.messages import trim_messages

trimmer = trim_messages(max_tokens=4000, strategy="last")
```

1주차의 "도구 무한 호출" 문제와 같은 근본 원인 — 컨텍스트에 불필요한 정보가 쌓여 토큰 낭비.

---

## 4. 액션 아이템

- [ ] `real_estate_graph.py`: `messages: list` → `Annotated[list[AnyMessage], add_messages]`
- [ ] 3주차 목요일: SqliteSaver 연동 + 메시지 트리밍 구현
- [ ] MCP 서버는 필요할 때만 연결/해제하여 고정 토큰 비용 절약
