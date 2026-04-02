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
    """부동산 실거래가 분석 에이전트를 생성합니다.

    Args:
        checkpointer: 대화 상태 저장소. None이면 InMemorySaver 사용.

    Returns:
        LangChain Agent 인스턴스 (astream 호출 가능)
    """

    # [Step 4] 모델 초기화
    model = ChatOpenAI(
        model=settings.OPENAI_MODEL,
        api_key=settings.OPENAI_API_KEY,
        temperature=0.1,
        max_tokens=4096,
        streaming=True,
    )

    # [Step 6] 메모리 (대화 상태 유지)
    if checkpointer is None:
        checkpointer = InMemorySaver()

    # [Step 7] 에이전트 조립
    agent = create_agent(
        model=model,
        tools=[search_apartment_trades, search_apartment_rentals, calculate_jeonse_ratio, search_pdf_reports],
        system_prompt=get_system_prompt(),
        response_format=ToolStrategy(ChatResponse),
        checkpointer=checkpointer,
    )

    return agent
