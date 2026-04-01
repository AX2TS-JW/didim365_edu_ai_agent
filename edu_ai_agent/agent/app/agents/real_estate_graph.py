"""
3주차: LangGraph StateGraph 기반 부동산 실거래가 분석 에이전트.

기존 create_agent() 방식에서 StateGraph로 전환.
- 코드가 흐름을 제어 (노드 + 엣지)
- LLM은 각 노드에서 맡은 역할만 수행
- 도구(tools.py)는 그대로 재사용
"""

from datetime import datetime
from typing import Literal
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver

from app.agents.prompts import get_system_prompt
from app.agents.tools import (
    search_apartment_trades,
    search_apartment_rentals,
    calculate_jeonse_ratio,
)
from app.core.config import settings


# ──────────────────────────────────────────────────────────
# State 정의 — 노드 간 전달되는 메모장
# ──────────────────────────────────────────────────────────

class AgentState(TypedDict):
    messages: list           # 대화 히스토리
    query_type: str          # "simple", "compare", "ambiguous", "jeonse_ratio"
    regions: list[str]       # ["강남구"] 또는 ["강남구", "송파구"]
    trade_type: str          # "매매", "전세", "전세가율"
    year_month: str          # 조회 년월 (YYYYMM)
    data: dict               # 조회 결과
    response: str            # 최종 답변


# ──────────────────────────────────────────────────────────
# LLM 초기화
# ──────────────────────────────────────────────────────────

def _get_llm():
    return ChatOpenAI(
        model=settings.OPENAI_MODEL,
        api_key=settings.OPENAI_API_KEY,
        temperature=0.1,
    )


# ──────────────────────────────────────────────────────────
# 노드 1: 질문 분석
# ──────────────────────────────────────────────────────────

_PARSE_PROMPT = """사용자의 부동산 질문을 분석하여 아래 JSON 형식으로만 답하세요. 다른 텍스트 없이 JSON만 출력하세요.

현재 날짜: {today}
현재 년월: {current_ym}

분석 기준:
- query_type: "simple"(단일 지역 조회), "compare"(2개 지역 비교), "jeonse_ratio"(전세가율/갈아타기), "ambiguous"(지역 없음/모호)
- regions: 기초자치단체(구/시/군) 이름 목록. 동 이름(판교, 잠실)이나 광역자치단체(서울, 부산)는 "ambiguous"로 분류
- trade_type: "매매", "전세", "전세가율"
- year_month: 조회할 년월 (YYYYMM). 미래 월은 현재 년월로 대체. "최근"이면 현재 년월

{{"query_type": "...", "regions": [...], "trade_type": "...", "year_month": "..."}}"""


def parse_query(state: AgentState) -> dict:
    """사용자 질문을 분석하여 query_type, regions, trade_type, year_month를 결정합니다."""
    llm = _get_llm()
    messages = state["messages"]
    user_msg = messages[-1].content if messages else ""

    today = datetime.now().strftime("%Y-%m-%d")
    current_ym = datetime.now().strftime("%Y%m")

    response = llm.invoke([
        SystemMessage(content=_PARSE_PROMPT.format(today=today, current_ym=current_ym)),
        HumanMessage(content=user_msg),
    ])

    # JSON 파싱
    import json
    try:
        text = response.content.strip()
        # ```json ... ``` 감싸진 경우 처리
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        parsed = json.loads(text)
    except (json.JSONDecodeError, IndexError):
        # 파싱 실패 시 ambiguous로 처리
        parsed = {
            "query_type": "ambiguous",
            "regions": [],
            "trade_type": "매매",
            "year_month": current_ym,
        }

    return {
        "query_type": parsed.get("query_type", "ambiguous"),
        "regions": parsed.get("regions", []),
        "trade_type": parsed.get("trade_type", "매매"),
        "year_month": parsed.get("year_month", current_ym),
    }


# ──────────────────────────────────────────────────────────
# 노드 2: 되묻기 (ambiguous일 때)
# ──────────────────────────────────────────────────────────

def ask_clarification(state: AgentState) -> dict:
    """모호한 질문에 대해 되묻는 응답을 생성합니다."""
    llm = _get_llm()
    messages = state["messages"]
    user_msg = messages[-1].content if messages else ""

    response = llm.invoke([
        SystemMessage(content=(
            "사용자의 부동산 질문이 모호합니다. "
            "기초자치단체(구/시/군) 단위의 지역명을 확인하는 짧은 되묻기 응답을 생성하세요. "
            "예: '어느 지역(구/시/군)의 시세를 조회할까요?'"
        )),
        HumanMessage(content=user_msg),
    ])

    return {"response": response.content}


# ──────────────────────────────────────────────────────────
# 노드 3: 데이터 조회 (단순)
# ──────────────────────────────────────────────────────────

def fetch_data_simple(state: AgentState) -> dict:
    """단일 지역 데이터를 조회합니다."""
    region = state["regions"][0] if state["regions"] else ""
    ym = state["year_month"]
    trade_type = state["trade_type"]

    if trade_type == "전세가율":
        result = calculate_jeonse_ratio.invoke({"region": region, "year_month": ym})
    elif trade_type == "전세":
        result = search_apartment_rentals.invoke({"region": region, "year_month": ym})
    else:
        result = search_apartment_trades.invoke({"region": region, "year_month": ym})

    return {"data": {"results": [{"region": region, "content": result}]}}


# ──────────────────────────────────────────────────────────
# 노드 4: 데이터 조회 (비교)
# ──────────────────────────────────────────────────────────

def fetch_data_compare(state: AgentState) -> dict:
    """2개 지역 데이터를 각각 조회합니다."""
    regions = state["regions"]
    ym = state["year_month"]
    trade_type = state["trade_type"]

    results = []
    for region in regions[:2]:  # 최대 2개
        if trade_type == "전세가율":
            result = calculate_jeonse_ratio.invoke({"region": region, "year_month": ym})
        elif trade_type == "전세":
            result = search_apartment_rentals.invoke({"region": region, "year_month": ym})
        else:
            result = search_apartment_trades.invoke({"region": region, "year_month": ym})
        results.append({"region": region, "content": result})

    return {"data": {"results": results}}


# ──────────────────────────────────────────────────────────
# 노드 5: 응답 생성
# ──────────────────────────────────────────────────────────

def generate_response(state: AgentState) -> dict:
    """조회 데이터를 바탕으로 최종 답변을 생성합니다."""
    llm = _get_llm()
    data = state.get("data", {})
    results = data.get("results", [])
    query_type = state.get("query_type", "simple")

    # 데이터를 텍스트로 조합
    data_text = ""
    for r in results:
        data_text += f"\n### {r['region']}\n{r['content']}\n"

    if query_type == "compare":
        instruction = "아래 두 지역의 데이터를 비교 분석하여 답변하세요. 가격 차이, 특징을 정리해주세요."
    else:
        instruction = "아래 데이터를 바탕으로 사용자에게 답변하세요. 핵심 데이터(총 건수, 가격 범위, 평균가)를 반드시 포함하세요."

    user_msg = state["messages"][-1].content if state["messages"] else ""

    response = llm.invoke([
        SystemMessage(content=f"{instruction}\n\n조회 데이터:\n{data_text}"),
        HumanMessage(content=user_msg),
    ])

    return {"response": response.content}


# ──────────────────────────────────────────────────────────
# 라우터: 질문 유형에 따라 다음 노드 결정
# ──────────────────────────────────────────────────────────

def route_by_query_type(state: AgentState) -> str:
    """query_type에 따라 다음 노드를 결정합니다."""
    qt = state.get("query_type", "ambiguous")
    if qt == "compare":
        return "fetch_compare"
    elif qt in ("simple", "jeonse_ratio"):
        return "fetch_simple"
    else:  # ambiguous + 예상 못한 값 → 되묻기 (안전)
        return "ask_clarification"


# ──────────────────────────────────────────────────────────
# 그래프 조립
# ──────────────────────────────────────────────────────────

def create_real_estate_graph(checkpointer=None):
    """부동산 실거래가 분석 StateGraph를 생성합니다."""

    graph = StateGraph(AgentState)

    # 노드 등록
    graph.add_node("parse", parse_query)
    graph.add_node("ask_clarification", ask_clarification)
    graph.add_node("fetch_simple", fetch_data_simple)
    graph.add_node("fetch_compare", fetch_data_compare)
    graph.add_node("respond", generate_response)

    # 엣지 연결
    graph.add_edge(START, "parse")
    graph.add_conditional_edges("parse", route_by_query_type, {
        "ask_clarification": "ask_clarification",
        "fetch_simple": "fetch_simple",
        "fetch_compare": "fetch_compare",
    })
    graph.add_edge("ask_clarification", END)
    graph.add_edge("fetch_simple", "respond")
    graph.add_edge("fetch_compare", "respond")
    graph.add_edge("respond", END)

    # 컴파일
    if checkpointer is None:
        checkpointer = InMemorySaver()

    return graph.compile(checkpointer=checkpointer)
