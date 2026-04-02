"""
3주차: LangGraph StateGraph 기반 부동산 실거래가 분석 에이전트.

기존 create_agent() 방식에서 StateGraph로 전환.
- 코드가 흐름을 제어 (노드 + 엣지)
- LLM은 각 노드에서 맡은 역할만 수행
- 도구(tools.py)는 그대로 재사용
"""

from datetime import datetime
from typing import Annotated, Literal
from typing_extensions import TypedDict

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AnyMessage, trim_messages
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
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
    messages: Annotated[list[AnyMessage], add_messages]  # 대화 히스토리 (reducer: 새 메시지 추가)
    intent: str              # "TOOL" (도구 호출 필요) 또는 "DIRECT" (LLM 직접 답변)
    query_type: str          # "simple", "compare", "ambiguous", "jeonse_ratio", "comprehensive"
    regions: list[str]       # ["강남구"] 또는 ["강남구", "송파구"]
    trade_type: str          # "매매", "전세", "전세가율"
    year_month: str          # 조회 년월 (YYYYMM)
    data: dict               # 조회 결과
    pdf_context: list[str]   # PDF 검색 결과 (하이브리드 검색)
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


# 멀티턴 메시지 트리밍: 대화가 길어지면 최근 메시지만 유지
MAX_MESSAGE_TOKENS = 4000

def _trim_messages(messages: list) -> list:
    """대화 히스토리가 너무 길면 최근 메시지만 유지합니다."""
    try:
        return trim_messages(
            messages,
            max_tokens=MAX_MESSAGE_TOKENS,
            strategy="last",
            token_counter=_get_llm(),
            allow_partial=False,
        )
    except Exception:
        # trim 실패 시 최근 10개만 유지 (폴백)
        return messages[-10:] if len(messages) > 10 else messages


# ──────────────────────────────────────────────────────────
# 노드 0: 인텐트 판별 — 도구 호출이 필요한가?
# ──────────────────────────────────────────────────────────

def intent_check(state: AgentState) -> dict:
    """도구 호출이 필요한 질문인지 LLM이 판단합니다."""
    llm = _get_llm()
    trimmed = _trim_messages(state["messages"])

    response = llm.invoke([
        SystemMessage(content=(
            "당신은 부동산 실거래가 AI 에이전트의 인텐트 분류기입니다.\n"
            "사용자의 질문을 보고 TOOL 또는 DIRECT 중 하나만 답하세요.\n\n"
            "TOOL: 새로운 실거래가 데이터 조회가 필요한 경우\n"
            "  - 특정 지역의 매매/전세 시세 조회\n"
            "  - 전세가율 계산\n"
            "  - 지역 비교\n"
            "  - 투자 판단 (PDF 검색 필요)\n\n"
            "DIRECT: 이전 대화 맥락만으로 답변 가능한 경우\n"
            "  - '자세히 보여줘', '정리해줘', '요약해줘'\n"
            "  - '더 알려줘', '목록으로 보여줘'\n"
            "  - 인사, 감사, 일반 대화\n"
            "  - 이전에 조회한 데이터에 대한 추가 질문\n\n"
            "TOOL 또는 DIRECT 중 하나만 답하세요."
        )),
        *trimmed,
    ])

    intent = "TOOL" if "TOOL" in response.content.upper() else "DIRECT"
    return {"intent": intent}


def direct_respond(state: AgentState) -> dict:
    """도구 호출 없이 이전 대화 맥락만으로 LLM이 직접 답변합니다."""
    llm = _get_llm()
    trimmed = _trim_messages(state["messages"])

    # 공통 서식 규칙
    format_rule = (
        "답변에 마크다운 문법(**, ##, ###, - 목록 등)을 절대 사용하지 마세요. "
        "이모지 섹션 헤더와 구분선(────), ▸ 기호로 깔끔하게 정리하세요. "
        "이전 대화에서 조회된 데이터가 있다면 그 데이터를 활용하여 답변하세요."
    )

    response = llm.invoke([
        SystemMessage(content=(
            "부동산 실거래가 AI 에이전트입니다. "
            "이전 대화 맥락을 참고하여 사용자의 질문에 답변하세요. "
            "새로운 데이터 조회 없이, 이미 대화에 있는 정보를 활용하세요. "
            f"{format_rule}"
        )),
        *trimmed,
    ])

    return {"response": response.content, "messages": [response]}


# ──────────────────────────────────────────────────────────
# 노드 1: 질문 분석
# ──────────────────────────────────────────────────────────

_PARSE_PROMPT = """사용자의 부동산 질문을 분석하여 아래 JSON 형식으로만 답하세요. 다른 텍스트 없이 JSON만 출력하세요.

현재 날짜: {today}
현재 년월: {current_ym}

분석 기준:
- query_type: "simple"(단일 지역 조회), "compare"(2개 지역 비교), "jeonse_ratio"(전세가율/갈아타기), "comprehensive"(투자 판단/전망/사도될까 등 종합 분석 필요), "ambiguous"(지역 없음/모호)
- regions: 사용자가 직접 말한 기초자치단체(구/시/군) 이름만 사용. 절대 동 이름을 시군구로 자체 변환하지 마세요. "개포동"→"강남구", "잠실"→"송파구" 이런 변환 금지. 동 이름이 들어오면 regions를 비우고 "ambiguous"로 분류
- 이전 대화 맥락: 현재 질문에 지역이 없지만 이전 대화에서 기초자치단체를 조회한 적이 있다면, 그 지역을 regions에 사용하세요. 예: 이전에 "강남구 매매"를 조회했고 현재 "전세 시세"만 물으면 → regions: ["강남구"]
- "자세히", "목록", "전부 보여줘", "상세", "더 보여줘" 같은 요청은 이전에 조회한 지역과 거래타입으로 다시 조회하세요. ambiguous로 분류하지 마세요.
- trade_type: "매매", "전세", "전세가율"
- year_month: 조회할 년월 (YYYYMM). 미래 월은 현재 년월로 대체. "최근"이면 현재 년월

{{"query_type": "...", "regions": [...], "trade_type": "...", "year_month": "..."}}"""


def parse_query(state: AgentState) -> dict:
    """사용자 질문을 분석하여 query_type, regions, trade_type, year_month를 결정합니다."""
    llm = _get_llm()
    messages = state["messages"]

    today = datetime.now().strftime("%Y-%m-%d")
    current_ym = datetime.now().strftime("%Y%m")

    # 이전 대화 맥락 포함 (트리밍으로 토큰 제한) — "거기서", "아까" 같은 후속 질문 지원
    trimmed = _trim_messages(messages)

    response = llm.invoke([
        SystemMessage(content=_PARSE_PROMPT.format(today=today, current_ym=current_ym)),
        *trimmed,
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

    # 코드 레벨 검증: regions가 실제 기초자치단체인지 확인
    from app.agents.tools import _resolve_region_code_fallback
    regions = parsed.get("regions", [])
    validated_regions = []
    for r in regions:
        if _resolve_region_code_fallback(r) is not None:
            validated_regions.append(r)

    # 검증 통과한 지역이 없으면 ambiguous로 강제
    query_type = parsed.get("query_type", "ambiguous")
    if query_type != "ambiguous" and not validated_regions:
        query_type = "ambiguous"

    return {
        "query_type": query_type,
        "regions": validated_regions,
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
            "사용자의 부동산 질문에서 기초자치단체(구/시/군) 단위의 지역을 확인해야 합니다. "
            "아래 규칙에 따라 짧은 되묻기 응답을 생성하세요:\n"
            "1) 동/읍/면 이름인 경우: 해당 동이 속한 기초자치단체를 알려주고 확인을 요청하세요. "
            "예: '개포동은 강남구에 속합니다. 강남구 매매 시세를 조회할까요?'\n"
            "2) 광역자치단체(서울, 부산 등)인 경우: 어느 구/시/군인지 되물어보세요. "
            "예: '서울의 어느 구를 조회할까요? (예: 강남구, 서초구, 송파구)'\n"
            "3) 지역이 전혀 없는 경우: '어느 지역(구/시/군)의 시세를 조회할까요?'"
        )),
        HumanMessage(content=user_msg),
    ])

    return {"response": response.content, "messages": [response]}


# ──────────────────────────────────────────────────────────
# 노드 3: 데이터 조회 (단순)
# ──────────────────────────────────────────────────────────

def _summarize_for_comprehensive(result: str) -> str:
    """comprehensive 질문용으로 도구 결과를 요약합니다. 요약 통계 + 상위 3건만 포함."""
    lines = result.split("\n")
    summary_lines = []
    for line in lines:
        # 요약 통계 줄, 헤더, 경고 줄은 유지
        if line.startswith("📊") or line.startswith("■") or line.startswith("⚠️"):
            summary_lines.append(line)
        # 개별 거래는 상위 3건만
        elif line.startswith("- "):
            if sum(1 for l in summary_lines if l.startswith("- ")) < 3:
                summary_lines.append(line)
    if not summary_lines:
        return result[:500]
    return "\n".join(summary_lines)


def fetch_data_simple(state: AgentState) -> dict:
    """단일 지역 데이터를 조회합니다."""
    region = state["regions"][0] if state["regions"] else ""
    ym = state["year_month"]
    trade_type = state["trade_type"]
    is_comprehensive = state.get("query_type") == "comprehensive"

    if trade_type == "전세가율":
        result = calculate_jeonse_ratio.invoke({"region": region, "year_month": ym})
    elif trade_type == "전세":
        result = search_apartment_rentals.invoke({"region": region, "year_month": ym})
    else:
        result = search_apartment_trades.invoke({"region": region, "year_month": ym})

    # comprehensive: 요약본 전달 (토큰 절약, 응답 속도 향상)
    content = _summarize_for_comprehensive(result) if is_comprehensive else result

    return {"data": {"results": [{"region": region, "content": content}]}}


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
# 노드 5: PDF 검색 (종합 판단용 — 하이브리드 검색)
# ──────────────────────────────────────────────────────────

def search_pdf(state: AgentState) -> dict:
    """ES에서 부동산 PDF 리포트를 하이브리드 검색합니다."""
    user_msg = state["messages"][-1].content if state["messages"] else ""
    regions = state.get("regions", [])

    # 검색 쿼리: 사용자 질문 + 지역명
    query = user_msg
    if regions:
        query = f"{' '.join(regions)} {user_msg}"

    try:
        from pipeline.search import search_hybrid
        results = search_hybrid(query, top_k=3)

        pdf_contexts = []
        for r in results:
            source = r.get("source_file", "unknown")
            page = r.get("page", "?")
            content = r.get("content", "")
            pdf_contexts.append(f"[{source} p.{page}] {content}")

        return {"pdf_context": pdf_contexts}
    except Exception as e:
        print(f"  ⚠️ PDF 검색 실패: {e}")
        return {"pdf_context": []}


# ──────────────────────────────────────────────────────────
# 노드 6: 응답 생성
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

    # PDF 컨텍스트 (종합 판단 시)
    pdf_context = state.get("pdf_context", [])
    pdf_text = ""
    if pdf_context:
        pdf_text = "\n\n### 참고 리포트 (PDF)\n" + "\n".join(pdf_context)

    # 공통 서식 규칙: 마크다운 금지, 이모지+구분선으로 시각화 (브라우저 텍스트 렌더링 호환)
    format_rule = (
        "답변에 마크다운 문법(**, ##, ###, - 목록 등)을 절대 사용하지 마세요. "
        "대신 아래 서식을 반드시 따르세요:\n"
        "1) 섹션 제목은 이모지로 시작 (예: 📊 실거래가 현황, 📈 시장 전망, 📋 정책 동향, 💡 종합 의견)\n"
        "2) 섹션 제목 아래에 구분선 ──────────────── 을 넣고, 섹션 사이에 빈 줄 2개를 넣어 여백을 확보\n"
        "3) 핵심 수치는 ▸ 기호를 사용 (예: 평균 매매가 ▸ 34.3억원)\n"
        "4) 긴 설명보다 짧은 문장으로 끊어서 작성\n"
        "5) 💡 종합 의견 섹션 맨 앞에 '✅ 결론: (한 줄 핵심 요약)'을 먼저 작성하고, 그 아래에 상세 설명\n"
        "6) 출처는 맨 아래에 📎 출처 섹션으로 모아주세요\n"
        "7) 데이터는 기초자치단체(구/시/군) 전체 데이터입니다. 특정 동만 필터링하지 말고 전체 데이터를 기반으로 답변하세요"
    )

    if query_type == "comprehensive":
        instruction = (
            "아래 실거래가 데이터와 전문가 리포트를 종합하여 사용자의 투자/매매 판단을 도와주세요. "
            "실거래가 수치, 시장 전망, 정책 동향을 모두 포함하여 균형 잡힌 분석을 제공하세요. "
            "출처(PDF 파일명)를 반드시 표기하세요. "
            f"{format_rule}"
        )
    elif query_type == "compare":
        instruction = f"아래 두 지역의 데이터를 비교 분석하여 답변하세요. 가격 차이, 특징을 정리해주세요. {format_rule}"
    else:
        instruction = f"아래 데이터를 바탕으로 사용자에게 답변하세요. 핵심 데이터(총 건수, 가격 범위, 평균가)를 반드시 포함하세요. {format_rule}"

    # 이전 대화 맥락 포함 (트리밍으로 토큰 제한)
    trimmed = _trim_messages(state["messages"])

    response = llm.invoke([
        SystemMessage(content=f"{instruction}\n\n조회 데이터:\n{data_text}{pdf_text}"),
        *trimmed,
    ])

    return {"response": response.content, "messages": [response]}


# ──────────────────────────────────────────────────────────
# 라우터: 질문 유형에 따라 다음 노드 결정
# ──────────────────────────────────────────────────────────

def route_by_query_type(state: AgentState) -> str:
    """query_type에 따라 다음 노드를 결정합니다."""
    qt = state.get("query_type", "ambiguous")
    if qt == "compare":
        return "fetch_compare"
    elif qt == "comprehensive":
        return "fetch_simple"  # 먼저 실거래가 조회 → 이후 search_pdf로 이동
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
    graph.add_node("intent", intent_check)
    graph.add_node("direct_respond", direct_respond)
    graph.add_node("parse", parse_query)
    graph.add_node("ask_clarification", ask_clarification)
    graph.add_node("fetch_simple", fetch_data_simple)
    graph.add_node("fetch_compare", fetch_data_compare)
    graph.add_node("search_pdf", search_pdf)
    graph.add_node("respond", generate_response)

    # 인텐트 분기: DIRECT → LLM 직접 답변, TOOL → 기존 StateGraph
    def route_by_intent(state: AgentState) -> str:
        return "direct_respond" if state.get("intent") == "DIRECT" else "parse"

    graph.add_edge(START, "intent")
    graph.add_conditional_edges("intent", route_by_intent, {
        "direct_respond": "direct_respond",
        "parse": "parse",
    })
    graph.add_edge("direct_respond", END)

    # 기존 경로
    graph.add_conditional_edges("parse", route_by_query_type, {
        "ask_clarification": "ask_clarification",
        "fetch_simple": "fetch_simple",
        "fetch_compare": "fetch_compare",
    })
    graph.add_edge("ask_clarification", END)

    # fetch 후 분기: comprehensive면 PDF 검색, 아니면 바로 응답
    def route_after_fetch(state: AgentState) -> str:
        if state.get("query_type") == "comprehensive":
            return "search_pdf"
        return "respond"

    graph.add_conditional_edges("fetch_simple", route_after_fetch, {
        "search_pdf": "search_pdf",
        "respond": "respond",
    })
    graph.add_edge("fetch_compare", "respond")
    graph.add_edge("search_pdf", "respond")
    graph.add_edge("respond", END)

    # 컴파일
    if checkpointer is None:
        checkpointer = InMemorySaver()

    return graph.compile(checkpointer=checkpointer)
