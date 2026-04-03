# Deep Agent 재귀 호출 구조 이슈 분석

> 4주차 Deep Agent 적용 과정에서 겪은 재귀 호출 제한 문제와 해결 과정을 정리합니다.

---

## 1. 문제 현상

"강남구 아파트 지금 사도 될까?" 같은 종합 판단 질문에서:
- `GraphRecursionError: Recursion limit of N reached` 반복 발생
- recursion_limit을 10 → 20 → 25 → 50으로 올려야 했음
- `MAX_TOOL_CALLS=5`에서도 강제 종료 발생

---

## 2. 원인 — ReAct vs Deep Agent의 내부 스텝 차이

### ReAct (3주차)

```
도구 호출 1회 = recursion 2스텝
  model 노드 (LLM이 도구 결정) → tool 노드 (도구 실행) → model 노드 (다음 판단)

5회 도구 호출 = ~10스텝 → recursion_limit=10으로 충분
```

### Deep Agent (4주차)

```
도구 호출 1회 = recursion 3~4스텝
  model 노드 (LLM이 도구 결정)
  → TodoListMiddleware.after_model (TODO 상태 체크)
  → tool 노드 (도구 실행)
  → model 노드 (다음 판단)

7회 도구 호출 = ~28스텝 → recursion_limit=25로 부족!
```

**Deep Agent는 SDK 내장 미들웨어(TodoListMiddleware 등)가 매 스텝마다 개입**하여 recursion 소모가 ReAct 대비 1.5~2배 많음.

---

## 3. 추가 문제 — MAX_TOOL_CALLS와 recursion_limit의 이중 제한

두 개의 제한이 독립적으로 동작하면서 충돌:

```
MAX_TOOL_CALLS = 15  (도구 호출 횟수 제한, agent_service.py)
recursion_limit = 50 (그래프 전체 스텝 제한, config.py)

문제 시나리오:
  도구 7회 호출 (MAX_TOOL_CALLS 기준 아직 여유)
  × 4스텝/도구 = 28스텝 (recursion_limit 25면 초과!)
  → MAX_TOOL_CALLS는 여유 있는데 recursion_limit에 먼저 걸림
```

### 관계 공식

```
필요한 recursion_limit ≥ MAX_TOOL_CALLS × (스텝/도구) + 여유
                       ≥ 15 × 4 + 10
                       ≥ 70

현재 설정: recursion_limit=50, MAX_TOOL_CALLS=15
→ 도구 ~10회까지 안전, 15회 도달 전에 recursion_limit 걸릴 수 있음
```

---

## 4. AGENT_MODE 기본값 불일치 버그

가장 찾기 어려웠던 버그:

```python
# chat.py
_AGENT_MODE = os.getenv("AGENT_MODE", "deep")     # 기본: deep

# agent_service.py
agent_mode = os.getenv("AGENT_MODE", "react")      # 기본: react ← 다름!
```

.env에 AGENT_MODE를 안 넣으면:
- chat.py는 "deep"으로 판단 → AgentService 사용
- agent_service.py는 "react"로 판단 → **ReAct 에이전트 생성 + MAX_TOOL_CALLS=5**

결과: Deep Agent라고 생각했는데 실제로는 ReAct가 5회 제한으로 동작.

---

## 5. 체크포인트 정합성 문제

MAX_TOOL_CALLS 강제 종료 시 체크포인트에 "미완료 tool_calls"가 남는 문제:

```
1번째 요청:
  LLM → tool_calls: [search_trades] → 체크포인트에 저장
  → MAX_TOOL_CALLS 도달 → 강제 종료
  → 체크포인트: tool_calls는 있는데 ToolMessage 없음 (깨진 상태)

2번째 요청 (같은 thread_id):
  체크포인트에서 이전 상태 로드 → tool_calls에 대한 ToolMessage 없음
  → OpenAI 400 에러: "tool_calls must be followed by tool messages"
```

### 해결: 미완료 tool_calls 패치 (정답지 방식)

```python
# 에러 발생 시:
state = await self.agent.aget_state(config)
messages = state.values.get("messages", [])

if messages[-1].tool_calls:  # 미완료 tool_calls 발견
    patches = []
    for tc in messages[-1].tool_calls:
        patches.append(ToolMessage(
            content="이전 요청 시간 초과로 결과 없음",
            tool_call_id=tc["id"]
        ))
    await self.agent.aupdate_state(config, {"messages": patches})
```

---

## 6. 프론트엔드 SSE 재연결 문제

fetchEventSource 라이브러리의 기본 동작이 "연결 끊기면 자동 재연결":

```
1번째 요청 → 응답 완료 → SSE 스트림 종료
→ fetchEventSource: "연결이 끊겼다!" → 자동 재연결
→ 같은 thread_id + 같은 message로 2번째 요청 전송
→ 체크포인트 충돌 → 400 에러 → 또 재연결 → 무한 반복
```

### 해결: step=done 시 AbortController로 즉시 종료

```typescript
onmessage(event) {
    const data = JSON.parse(event.data);
    onChunk(step, content, metadata, toolCalls, name);

    if(step === 'done') {
        controller.abort();  // 응답 완료 → SSE 연결 즉시 종료
    }
},
onclose() {
    controller.abort();      // 서버 연결 닫힘 → 재연결 안 함
},
onerror(err) {
    controller.abort();
    throw err;               // 에러 시 재연결 완전 중단
}
```

---

## 7. 설정 값 변경 이력

| 시점 | recursion_limit | MAX_TOOL_CALLS | 이유 |
|------|----------------|----------------|------|
| 1주차 | 10 | 3 | ReAct 기본 |
| 2주차 | 10 | 5 | 비교 질문 대응 |
| 4주차 Day 1 | 10 → 20 | 5 | Deep Agent 스텝 증가 |
| 4주차 Day 5 | 20 → 25 | 5 | 여전히 부족 |
| 4주차 Day 5 | 25 → 50 | 5 → 15 | AGENT_MODE 기본값 수정 후 |

---

## 8. 교훈

1. **SDK 도입 시 내부 동작을 이해해야 함** — Deep Agent는 미들웨어가 매 스텝마다 개입하여 recursion 소모가 2배
2. **환경변수 기본값은 한 곳에서 관리** — 여러 파일에서 다른 기본값을 사용하면 디버깅이 극히 어려움
3. **이중 제한(MAX_TOOL_CALLS + recursion_limit)은 관계를 계산해야 함** — 독립적으로 설정하면 예상치 못한 곳에서 먼저 걸림
4. **체크포인트 정합성은 반드시 방어** — 중간 종료 시 미완료 상태가 남으면 후속 요청에서 연쇄 에러
5. **프론트엔드 SSE 라이브러리의 기본 동작을 확인** — 자동 재연결이 오히려 문제를 유발할 수 있음
