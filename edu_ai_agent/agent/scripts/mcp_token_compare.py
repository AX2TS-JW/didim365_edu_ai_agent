"""MCP 토큰 비용 비교 테스트

같은 질문("StateGraph add_node 파라미터 설명해줘")을 두 가지 방식으로 호출:
1. 도구 없이 LLM 기존 지식만
2. MCP 검색 결과를 컨텍스트에 포함하여 호출

OpenAI API의 usage 필드로 실제 토큰 소비량을 비교합니다.
"""

import json
import os
import sys

from openai import OpenAI

# ── 설정 ──────────────────────────────────────────────
QUESTION = "LangGraph StateGraph의 add_node 파라미터를 설명해줘"

# MCP 검색 결과를 시뮬레이션 (실제 MCP 서버에서 반환된 내용 축약)
MCP_SEARCH_RESULT = """
[MCP docs-langchain 검색 결과]

1. Nodes (Python langgraph/graph-api)
In LangGraph, nodes are typically functions (sync or async) that accept the following arguments:
- state: The state of the graph
- config: A RunnableConfig object that contains configuration information like thread_id and tracing information like tags
You can add nodes to a graph using the add_node method.
Behind the scenes, functions are converted to RunnableLambda, which add batch and async support to your function, along with native tracing and debugging.
If you add a node to a graph without specifying a name, it will be given a default name equivalent to the function name.

2. Nodes (JavaScript langgraph/graph-api)
In LangGraph, nodes are typically functions (sync or async) that accept the following arguments: state, config.
You can add nodes to a graph using the addNode method. For better type safety, use the GraphNode type utility or State.Node to type your node functions.

3. Add a subgraph as a node (Python)
You can add a compiled subgraph directly to add_node. No wrapper function is needed.

4. Graph compilation code (thinking-in-langgraph)
Create the graph: workflow = StateGraph(EmailAgentState)
Add nodes with workflow.add_node("agent", call_model)

5. Extended example: specifying model and system message at runtime
from langgraph.graph import END, MessagesState, StateGraph, START
"""

# MCP 도구 스키마 (tools/list로 받아오는 것과 동일)
MCP_TOOL_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_docs_by_lang_chain",
            "description": "Search across the Docs by LangChain knowledge base to find relevant information, code examples, API references, and guides. Use this tool when you need to answer questions about Docs by LangChain, find specific documentation, understand how features work, or locate implementation details.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_page_docs_by_lang_chain",
            "description": "Retrieve the full content of a specific documentation page from Docs by LangChain by its path. Use this tool when you already know the page path and need the complete content of that page rather than just a snippet.",
            "parameters": {
                "type": "object",
                "properties": {
                    "page": {
                        "type": "string",
                        "description": "The page path to retrieve",
                    }
                },
                "required": ["page"],
            },
        },
    },
]


def call_without_mcp(client: OpenAI, model: str) -> dict:
    """테스트 1: MCP 없이 — LLM 기존 지식만으로 답변"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "너는 LangGraph/LangChain 전문가야. 한국어로 답변해."},
            {"role": "user", "content": QUESTION},
        ],
        temperature=0,
    )
    return {
        "label": "MCP 미사용 (기존 지식)",
        "answer": response.choices[0].message.content,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        },
    }


def call_with_mcp_tool_only(client: OpenAI, model: str) -> dict:
    """테스트 2: MCP 도구 스키마만 등록 (호출 안 함)"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "너는 LangGraph/LangChain 전문가야. 한국어로 답변해."},
            {"role": "user", "content": QUESTION},
        ],
        tools=MCP_TOOL_SCHEMA,
        temperature=0,
    )
    # tool_call을 요청할 수도 있지만, 토큰 측정이 목적
    msg = response.choices[0].message
    answer = msg.content or "(LLM이 도구 호출을 선택함 — tool_calls 발생)"
    return {
        "label": "MCP 도구 등록만 (스키마 비용)",
        "answer": answer,
        "tool_calls": bool(msg.tool_calls),
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        },
    }


def call_with_mcp_result(client: OpenAI, model: str) -> dict:
    """테스트 3: MCP 검색 결과를 컨텍스트에 포함"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "너는 LangGraph/LangChain 전문가야. 한국어로 답변해. 아래 검색 결과를 참고하여 정확하게 답변해."},
            {"role": "user", "content": f"{QUESTION}\n\n---\n참고 자료:\n{MCP_SEARCH_RESULT}"},
        ],
        temperature=0,
    )
    return {
        "label": "MCP 검색 결과 포함 (풀 비용)",
        "answer": response.choices[0].message.content,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.tokens,
        }
        if hasattr(response.usage, "tokens")
        else {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        },
    }


def main():
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", "gpt-4.1-mini")

    if not api_key:
        print("ERROR: OPENAI_API_KEY 환경변수가 필요합니다.")
        sys.exit(1)

    client = OpenAI(api_key=api_key)
    print(f"모델: {model}")
    print(f"질문: {QUESTION}")
    print("=" * 70)

    results = []

    # 테스트 1
    print("\n[1/3] MCP 미사용 호출 중...")
    r1 = call_without_mcp(client, model)
    results.append(r1)
    print(f"  → prompt: {r1['usage']['prompt_tokens']}, completion: {r1['usage']['completion_tokens']}, total: {r1['usage']['total_tokens']}")

    # 테스트 2
    print("\n[2/3] MCP 도구 스키마만 등록하여 호출 중...")
    r2 = call_with_mcp_tool_only(client, model)
    results.append(r2)
    print(f"  → prompt: {r2['usage']['prompt_tokens']}, completion: {r2['usage']['completion_tokens']}, total: {r2['usage']['total_tokens']}")
    if r2.get("tool_calls"):
        print("  ⚠ LLM이 도구 호출을 선택했습니다 (답변 대신 tool_call 반환)")

    # 테스트 3
    print("\n[3/3] MCP 검색 결과 포함하여 호출 중...")
    r3 = call_with_mcp_result(client, model)
    results.append(r3)
    print(f"  → prompt: {r3['usage']['prompt_tokens']}, completion: {r3['usage']['completion_tokens']}, total: {r3['usage']['total_tokens']}")

    # ── 비교 리포트 ──────────────────────────────────────
    print("\n" + "=" * 70)
    print("📊 토큰 비교 리포트")
    print("=" * 70)

    baseline = results[0]["usage"]["prompt_tokens"]

    for r in results:
        u = r["usage"]
        diff = u["prompt_tokens"] - baseline
        diff_str = f"(+{diff})" if diff > 0 else "(기준)"
        print(f"\n🔹 {r['label']}")
        print(f"   입력 토큰:  {u['prompt_tokens']:>6} {diff_str}")
        print(f"   출력 토큰:  {u['completion_tokens']:>6}")
        print(f"   총 토큰:    {u['total_tokens']:>6}")

    # 증가율
    if baseline > 0:
        schema_increase = results[1]["usage"]["prompt_tokens"] - baseline
        full_increase = results[2]["usage"]["prompt_tokens"] - baseline
        print(f"\n📈 도구 스키마만 등록 시 입력 토큰 증가: +{schema_increase} ({schema_increase/baseline*100:.1f}%)")
        print(f"📈 검색 결과까지 포함 시 입력 토큰 증가: +{full_increase} ({full_increase/baseline*100:.1f}%)")

    # 답변 전문
    print("\n" + "=" * 70)
    print("📝 답변 비교 (전문)")
    print("=" * 70)
    for r in results:
        print(f"\n{'─' * 40}")
        print(f"🔹 {r['label']}:")
        print(f"{'─' * 40}")
        print(r["answer"])


if __name__ == "__main__":
    main()
