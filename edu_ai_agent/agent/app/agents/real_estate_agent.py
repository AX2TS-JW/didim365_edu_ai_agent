"""
부동산 실거래가 분석 에이전트.
LangChain create_agent()를 사용하여 에이전트를 생성합니다.

[LangChain 매핑]
- https://docs.langchain.com/oss/python/langchain/quickstart → Step 4~7 (Model + ResponseFormat + Memory + Assembly)
- https://docs.langchain.com/oss/python/langchain/agents → Creating Agents, Structured Output > ToolStrategy
"""

from dataclasses import dataclass, field

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from app.agents.prompts import get_system_prompt
from langchain.tools import tool

from app.agents.tools import search_apartment_trades, search_apartment_rentals, calculate_jeonse_ratio
from app.core.config import settings


# ──────────────────────────────────────────────────────────
# PDF 검색 서브에이전트 (StateGraph → @tool 래핑)
# ──────────────────────────────────────────────────────────

from typing_extensions import TypedDict
from langgraph.graph import StateGraph as SubStateGraph, START as SUB_START, END as SUB_END


class SearchState(TypedDict):
    query: str
    bm25_hits: list[dict]
    vector_hits: list[dict]
    merged_hits: list[dict]
    result: str


def _bm25_search(state: SearchState) -> dict:
    """BM25 키워드 검색"""
    try:
        from pipeline.search import search_bm25
        hits = search_bm25(state["query"], top_k=5)
        return {"bm25_hits": hits}
    except Exception:
        return {"bm25_hits": []}


def _vector_search(state: SearchState) -> dict:
    """kNN 벡터 검색"""
    try:
        from pipeline.search import search_vector
        hits = search_vector(state["query"], top_k=5)
        return {"vector_hits": hits}
    except Exception:
        return {"vector_hits": []}


def _merge_results(state: SearchState) -> dict:
    """BM25 + Vector 결과 병합 + 중복 제거"""
    seen = set()
    merged = []
    for hit in state["bm25_hits"] + state["vector_hits"]:
        key = f"{hit.get('source_file', '')}_{hit.get('page', '')}"
        if key not in seen:
            seen.add(key)
            merged.append(hit)
    # score 내림차순 정렬
    merged.sort(key=lambda x: x.get("score", 0), reverse=True)
    return {"merged_hits": merged[:5]}


def _rerank_results(state: SearchState) -> dict:
    """HuggingFace cross-encoder로 리랭킹"""
    hits = state["merged_hits"]
    query = state["query"]

    if not hits or len(hits) <= 1:
        return {"merged_hits": hits}

    try:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        pairs = [(query, hit.get("content", "")) for hit in hits]
        scores = model.predict(pairs)

        # 리랭킹 점수로 재정렬
        reranked = sorted(zip(hits, scores), key=lambda x: x[1], reverse=True)
        reranked_hits = [hit for hit, score in reranked[:5]]

        return {"merged_hits": reranked_hits}
    except Exception as e:
        # 리랭킹 실패 시 기존 결과 유지 (graceful fallback)
        print(f"  ⚠️ 리랭킹 실패 (score 정렬 유지): {e}")
        return {"merged_hits": hits}


def _format_search_result(state: SearchState) -> dict:
    """검색 결과를 텍스트로 포맷"""
    hits = state["merged_hits"]
    if not hits:
        return {"result": "관련 PDF 보고서를 찾을 수 없습니다."}

    contexts = []
    for r in hits:
        source = r.get("source_file", "unknown")
        page = r.get("page", "?")
        content = r.get("content", "")
        contexts.append(f"[{source} p.{page}] {content}")

    return {"result": "\n\n".join(contexts)}


# 서브에이전트 그래프 조립 (BM25 + Vector 병렬 → merge → rerank → format)
_search_graph = SubStateGraph(SearchState)
_search_graph.add_node("bm25", _bm25_search)
_search_graph.add_node("vector", _vector_search)
_search_graph.add_node("merge", _merge_results)
_search_graph.add_node("rerank", _rerank_results)
_search_graph.add_node("format", _format_search_result)

# 병렬 fan-out: START → bm25 + vector 동시 실행
_search_graph.add_edge(SUB_START, "bm25")
_search_graph.add_edge(SUB_START, "vector")
# fan-in: bm25 + vector → merge → rerank → format
_search_graph.add_edge("bm25", "merge")
_search_graph.add_edge("vector", "merge")
_search_graph.add_edge("merge", "rerank")
_search_graph.add_edge("rerank", "format")
_search_graph.add_edge("format", SUB_END)

_compiled_search = _search_graph.compile()


@tool
def search_pdf_reports(query: str) -> str:
    """부동산 관련 PDF 보고서(KB부동산, 국토연구원, 한국은행 등)에서 시장 전망, 정책 동향, 투자 분석 정보를 검색합니다.
    실거래가 데이터로 알 수 없는 전문가 분석, 정책 변화, 시장 전망이 필요할 때 사용하세요.

    Args:
        query: 검색할 키워드 (예: "강남구 시장 전망", "금리 인상 영향", "전세 시장 동향")

    Returns:
        관련 PDF 보고서 내용 (상위 5건, BM25+Vector 병렬 검색)
    """
    try:
        result = _compiled_search.invoke({
            "query": query,
            "bm25_hits": [],
            "vector_hits": [],
            "merged_hits": [],
            "result": "",
        })
        return result["result"]
    except Exception as e:
        return f"PDF 검색 중 오류: {e}"


# [Step 5] 응답 포맷 정의
# agent_service.py가 tool_calls에서 "ChatResponse"라는 이름의 tool_call을 파싱하므로
# 클래스명을 반드시 ChatResponse로 맞춰야 합니다.
@dataclass
class ChatResponse:
    """에이전트의 구조화된 응답 포맷.
    프론트엔드의 SSE 이벤트 파싱과 일치해야 합니다.
    """
    message_id: str     # UUID 형식
    content: str        # 사용자에게 보여줄 답변 텍스트
    metadata: dict = field(default_factory=dict)  # sql, data, chart 등 부가 정보


def create_real_estate_agent(checkpointer=None):
    """부동산 실거래가 분석 에이전트를 생성합니다. (ReAct 모드)"""

    model = ChatOpenAI(
        model=settings.OPENAI_MODEL,
        api_key=settings.OPENAI_API_KEY,
        temperature=0.1,
        max_tokens=4096,
        streaming=True,
    )

    if checkpointer is None:
        checkpointer = InMemorySaver()

    agent = create_agent(
        model=model,
        tools=[search_apartment_trades, search_apartment_rentals, calculate_jeonse_ratio, search_pdf_reports],
        system_prompt=get_system_prompt(),
        response_format=ToolStrategy(ChatResponse),
        checkpointer=checkpointer,
    )

    return agent


def create_deep_real_estate_agent(checkpointer=None):
    """부동산 실거래가 분석 Deep Agent를 생성합니다. (4주차)

    create_agent 대비 추가 기능:
    - write_todos: 복합 질문을 단계별로 분해
    - VFS: 중간 결과를 파일로 저장 (토큰 절약)
    - task(): 서브에이전트에게 작업 위임 (컨텍스트 격리)
    - 서브에이전트 3개: DataCollector, Analyst, Reporter
    """
    from deepagents import create_deep_agent, SubAgent

    model = ChatOpenAI(
        model=settings.OPENAI_MODEL,
        api_key=settings.OPENAI_API_KEY,
        temperature=0.1,
        max_tokens=4096,
        streaming=True,
    )

    if checkpointer is None:
        checkpointer = InMemorySaver()

    # 서브에이전트 3개 정의 (VFS로 데이터 공유)
    data_collector: SubAgent = {
        "name": "data-collector",
        "description": "부동산 실거래가 데이터를 수집하고 VFS에 저장합니다.",
        "system_prompt": (
            "당신은 부동산 데이터 수집 전문가입니다. "
            "주어진 지역과 조건에 맞는 실거래가 데이터를 조회하고, "
            "필요 시 PDF 보고서에서 관련 정보를 검색합니다.\n\n"
            "중요: 수집한 데이터는 반드시 write_file로 VFS에 저장하세요.\n"
            "- 매매 데이터: write_file('/data/{지역명}_매매.md', 결과)\n"
            "- 전세 데이터: write_file('/data/{지역명}_전세.md', 결과)\n"
            "- PDF 검색: write_file('/data/{지역명}_전망.md', 결과)\n"
            "이렇게 저장하면 analyst와 reporter가 read_file로 읽어서 활용합니다."
        ),
        "tools": [search_apartment_trades, search_apartment_rentals, search_pdf_reports],
    }

    analyst: SubAgent = {
        "name": "analyst",
        "description": "VFS에 저장된 데이터를 읽어서 분석합니다.",
        "system_prompt": (
            "당신은 부동산 데이터 분석 전문가입니다.\n\n"
            "중요: 먼저 read_file 또는 ls로 VFS에 저장된 데이터를 확인하세요.\n"
            "- ls('/data/')로 수집된 파일 목록 확인\n"
            "- read_file('/data/{지역명}_매매.md')로 데이터 읽기\n"
            "VFS에 데이터가 있으면 도구 재호출 없이 파일에서 읽어서 분석하세요.\n"
            "VFS에 없는 데이터만 도구로 직접 조회하세요.\n\n"
            "분석 결과는 write_file('/reports/{분석명}.md')로 저장하세요.\n"
            "분석 결과에는 반드시 수치 근거를 포함하세요."
        ),
        "tools": [calculate_jeonse_ratio, search_apartment_trades, search_apartment_rentals],
    }

    reporter: SubAgent = {
        "name": "reporter",
        "description": "VFS에 저장된 분석 결과를 읽어서 종합 리포트를 작성합니다.",
        "system_prompt": (
            "당신은 부동산 리포트 작성 전문가입니다.\n\n"
            "중요: 먼저 ls로 VFS 파일 목록을 확인하고, read_file로 데이터와 분석 결과를 읽으세요.\n"
            "- ls('/data/')와 ls('/reports/')로 파일 확인\n"
            "- 수집된 데이터와 분석 결과를 종합하여 리포트 작성\n\n"
            "서식 규칙:\n"
            "- 이모지 섹션 헤더(📊, 📈, 💡), 구분선(────), ▸ 수치 표기\n"
            "- 마크다운 문법(**, ##) 사용 금지\n"
            "- 출처(PDF 파일명, 데이터 기준일) 반드시 표기\n"
            "- ✅ 결론 한 줄을 맨 앞에"
        ),
    }

    # 장기 기억: AGENTS.md에서 도메인 지식 로드
    import os
    memory_path = os.path.join(os.path.dirname(__file__), "AGENTS.md")
    memory_files = [memory_path] if os.path.exists(memory_path) else None

    agent = create_deep_agent(
        model=model,
        tools=[search_apartment_trades, search_apartment_rentals, calculate_jeonse_ratio, search_pdf_reports],  # 단순 질문은 직접, 복합은 task()로 위임
        system_prompt=get_system_prompt(),
        response_format=ToolStrategy(ChatResponse),
        subagents=[data_collector, analyst, reporter],
        memory=memory_files,
        checkpointer=checkpointer,
    )

    return agent
