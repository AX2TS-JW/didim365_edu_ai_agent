"""
3주차: StateGraph 기반 AgentService.

기존 agent_service.py(ReAct)와 동일한 SSE 이벤트 형식을 유지하면서
내부적으로 StateGraph를 사용합니다.

SSE 이벤트 형식 (프론트엔드 호환):
  {"step": "model", "tool_calls": ["parse"]}
  {"step": "tools", "name": "fetch_simple", "content": "..."}
  {"step": "done", "message_id": "...", "content": "...", "metadata": {}, "created_at": "..."}
"""

import asyncio
import json
import uuid
from datetime import datetime

from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.agents.real_estate_graph import create_real_estate_graph, AgentState
from app.core.config import settings
from app.utils.logger import log_execution, custom_logger


class GraphAgentService:
    def __init__(self):
        self.graph = None
        self.checkpointer = None

    def _create_graph(self):
        """StateGraph 에이전트를 생성합니다."""
        if self.checkpointer is None:
            self.checkpointer = InMemorySaver()

        self.graph = create_real_estate_graph(checkpointer=self.checkpointer)

        # Opik 트레이싱 (선택)
        opik_settings = settings.OPIK
        if opik_settings and opik_settings.PROJECT:
            try:
                from opik.integrations.langchain import track_langgraph, OpikTracer
                tracer = OpikTracer(project_name=opik_settings.PROJECT)
                self.graph = track_langgraph(self.graph, opik_tracer=tracer)
                custom_logger.info(f"Opik 트레이싱 활성화: project={opik_settings.PROJECT}")
            except ImportError:
                custom_logger.info("Opik 미설치 — 트레이싱 비활성화")

    @log_execution
    async def process_query(self, user_messages: str, thread_id: uuid.UUID):
        """StateGraph 에이전트를 실행하고 SSE 이벤트를 스트리밍합니다."""
        try:
            # 그래프 초기화
            if self.graph is None:
                self._create_graph()

            custom_logger.info(f"[Graph] 사용자 메시지: {user_messages}")

            # 초기 상태
            initial_state = {
                "messages": [HumanMessage(content=user_messages)],
                "query_type": "",
                "regions": [],
                "trade_type": "",
                "year_month": "",
                "data": {},
                "pdf_context": [],
                "response": "",
            }

            config = {"configurable": {"thread_id": str(thread_id)}}

            # StateGraph 스트리밍 실행
            async for chunk in self.graph.astream(initial_state, config=config, stream_mode="updates"):
                custom_logger.info(f"[Graph] 청크: {list(chunk.keys())}")

                for node_name, node_output in chunk.items():
                    if node_name == "__start__":
                        continue

                    # parse 노드: 질문 분석 완료
                    if node_name == "parse":
                        qt = node_output.get("query_type", "")
                        regions = node_output.get("regions", [])
                        trade_type = node_output.get("trade_type", "")
                        custom_logger.info(f"[Graph] 분석: type={qt}, regions={regions}, trade={trade_type}")
                        yield json.dumps({
                            "step": "model",
                            "tool_calls": [f"분석: {qt} / {', '.join(regions) if regions else '미정'} / {trade_type}"]
                        }, ensure_ascii=False)

                    # ask_clarification 노드: 되묻기
                    elif node_name == "ask_clarification":
                        response = node_output.get("response", "")
                        yield json.dumps({
                            "step": "done",
                            "message_id": str(uuid.uuid4()),
                            "role": "assistant",
                            "content": response,
                            "metadata": {},
                            "created_at": datetime.utcnow().isoformat(),
                        }, ensure_ascii=False)

                    # fetch 노드: 데이터 조회 결과
                    elif node_name in ("fetch_simple", "fetch_compare"):
                        data = node_output.get("data", {})
                        results = data.get("results", [])
                        for r in results:
                            yield json.dumps({
                                "step": "tools",
                                "name": f"조회: {r.get('region', '')}",
                                "content": r.get("content", "")[:500],
                            }, ensure_ascii=False)

                    # search_pdf 노드: PDF 검색 결과
                    elif node_name == "search_pdf":
                        pdf_context = node_output.get("pdf_context", [])
                        if pdf_context:
                            yield json.dumps({
                                "step": "tools",
                                "name": "PDF 리포트 검색",
                                "content": f"{len(pdf_context)}건의 관련 리포트를 찾았습니다.",
                            }, ensure_ascii=False)

                    # respond 노드: 최종 답변
                    elif node_name == "respond":
                        response = node_output.get("response", "")
                        yield json.dumps({
                            "step": "done",
                            "message_id": str(uuid.uuid4()),
                            "role": "assistant",
                            "content": response,
                            "metadata": {},
                            "created_at": datetime.utcnow().isoformat(),
                        }, ensure_ascii=False)

        except Exception as e:
            import traceback
            custom_logger.error(f"[Graph] Error: {e}")
            custom_logger.error(traceback.format_exc())
            yield json.dumps({
                "step": "done",
                "message_id": str(uuid.uuid4()),
                "role": "assistant",
                "content": "처리 중 오류가 발생했습니다. 다시 시도해주세요.",
                "metadata": {},
                "created_at": datetime.utcnow().isoformat(),
                "error": str(e),
            }, ensure_ascii=False)
