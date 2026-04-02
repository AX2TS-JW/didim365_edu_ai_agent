import asyncio
import contextlib
from datetime import datetime
import json
from typing import Optional
import uuid

from app.utils.logger import log_execution, custom_logger

from langchain_core.messages import HumanMessage
from langgraph.errors import GraphRecursionError
from opik.integrations.langchain import track_langgraph, OpikTracer
from app.core.config import settings


class AgentService:
    def __init__(self):
        # IMP: LangChain을 통해 사용할 LLM(OpenAI) 객체 초기화 구현. 에이전트의 두뇌 역할을 합니다.
        self.agent = None
        self.checkpointer = None
        self.progress_queue: asyncio.Queue = asyncio.Queue()

    def _create_agent(self, thread_id: uuid.UUID = None):
        """LangChain 에이전트 생성"""
        # IMP: create_agent()를 사용하여 LangGraph 기반의 에이전트를 생성.
        # LLM 모델, 사용할 도구(Tools), 시스템 프롬프트, 상태 저장소(Checkpointer), 응답 포맷(ToolStrategy)을 결합.
        import os
        from langgraph.checkpoint.memory import InMemorySaver

        # 체크포인터는 한 번만 생성 (멀티턴 대화를 위해 재사용)
        if self.checkpointer is None:
            self.checkpointer = InMemorySaver()

        # AGENT_MODE=deep이면 Deep Agent, 아니면 기존 ReAct
        agent_mode = os.getenv("AGENT_MODE", "react")
        if agent_mode == "deep":
            from app.agents.real_estate_agent import create_deep_real_estate_agent
            agent = create_deep_real_estate_agent(checkpointer=self.checkpointer)
            custom_logger.info("[Deep Agent] create_deep_agent()로 생성")
        else:
            from app.agents.real_estate_agent import create_real_estate_agent
            agent = create_real_estate_agent(checkpointer=self.checkpointer)

        # Opik 트레이싱: 에이전트 실행 과정을 자동 기록
        opik_settings = settings.OPIK
        if opik_settings and opik_settings.PROJECT:
            tracer = OpikTracer(project_name=opik_settings.PROJECT)
            self.agent = track_langgraph(agent, opik_tracer=tracer)
            custom_logger.info(f"Opik 트레이싱 활성화: project={opik_settings.PROJECT}")
        else:
            self.agent = agent

    # 실제 대화 로직
    @log_execution
    async def process_query(self, user_messages: str, thread_id: uuid.UUID):
        """LangChain Messages 형식의 쿼리를 처리하고 AIMessage 형식으로 반환합니다."""
        try:
            # 에이전트 초기화 (한 번만)
            self._create_agent(thread_id=thread_id)

            custom_logger.info(f"사용자 메시지: {user_messages}")

            # Tool 호출 횟수 제한 (무한 루프 방지)
            # 비교/추론 질문은 2개 도구 + 재시도가 필요하므로 5회로 상향
            MAX_TOOL_CALLS = 5
            tool_call_count = 0
            tool_limit_reached = False  # 제한 도달 시 스트림을 끝까지 소비하기 위한 플래그

            # Checkpointer 상태 오류 시 자동 재시도
            MAX_RETRIES = 1
            retry_count = 0

            # IMP: LangGraph 에이전트에 사용자의 메시지를 HumanMessage 형태로 전달하고,
            # thread_id를 통해 대화 문맥(Context)을 유지하며 비동기 스트리밍(astream)으로 실행하는 구현.
            agent_stream = self.agent.astream(
                {"messages": [HumanMessage(content=user_messages)]},
                config={
                    "configurable": {"thread_id": str(thread_id)},
                    "recursion_limit": settings.DEEPAGENT_RECURSION_LIMIT,
                },
                stream_mode="updates",
            )

            agent_iterator = agent_stream.__aiter__()
            agent_task = asyncio.create_task(agent_iterator.__anext__())
            progress_task = asyncio.create_task(self.progress_queue.get())

            while True:
                pending = {agent_task}
                if progress_task is not None:
                    pending.add(progress_task)

                done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)

                if progress_task in done:
                    try:
                        progress_event = progress_task.result()
                        yield json.dumps(progress_event, ensure_ascii=False)
                        progress_task = asyncio.create_task(self.progress_queue.get())
                    except asyncio.CancelledError:
                        progress_task = None
                    except Exception as e:
                        # progress_task에서 예외 발생 시 로그만 남기고 계속 진행
                        custom_logger.error(f"Error in progress_task: {e}")
                        progress_task = None

                if agent_task in done:
                    try:
                        chunk = agent_task.result()
                    except StopAsyncIteration:
                        agent_task = None
                        break
                    except Exception as e:
                        # Task에서 발생한 예외 처리
                        custom_logger.error(f"Error in agent_task: {e}")
                        import traceback
                        custom_logger.error(traceback.format_exc())

                        # tool_limit_reached 상태에서 발생한 에러는 무시
                        # (이미 강제 응답을 전송했으므로 에러 메시지로 덮어씌우지 않음)
                        if tool_limit_reached:
                            custom_logger.info("Tool 제한 후 스트림 소비 중 에러 — 무시 (이미 응답 전송됨)")
                            agent_task = None
                            break

                        # tool_calls/tool_call_id 불일치: checkpointer 상태 깨짐 → 초기화 후 재시도
                        error_str = str(e)
                        if "tool_calls" in error_str and "tool_call_id" in error_str and retry_count < MAX_RETRIES:
                            retry_count += 1
                            custom_logger.warning(
                                f"Corrupted thread state detected (thread={thread_id}). "
                                f"Resetting checkpointer and retrying ({retry_count}/{MAX_RETRIES})."
                            )
                            # checkpointer 초기화 → 다음 _create_agent에서 새로 생성
                            self.checkpointer = None
                            self._create_agent(thread_id=thread_id)
                            # 새 스트림으로 재시도
                            agent_stream = self.agent.astream(
                                {"messages": [HumanMessage(content=user_messages)]},
                                config={
                                    "configurable": {"thread_id": str(thread_id)},
                                    "recursion_limit": settings.DEEPAGENT_RECURSION_LIMIT,
                                },
                                stream_mode="updates",
                            )
                            agent_iterator = agent_stream.__aiter__()
                            agent_task = asyncio.create_task(agent_iterator.__anext__())
                            tool_call_count = 0
                            tool_limit_reached = False
                            continue

                        agent_task = None
                        # 에러를 스트리밍으로 전송
                        error_response = {
                            "step": "done",
                            "message_id": str(uuid.uuid4()),
                            "role": "assistant",
                            "content": "처리 중 오류가 발생했습니다. 다시 시도해주세요.",
                            "metadata": {},
                            "created_at": datetime.utcnow().isoformat(),
                            "error": str(e)
                        }
                        yield json.dumps(error_response, ensure_ascii=False)
                        break

                    custom_logger.info(f"에이전트 청크: {chunk}")
                    try:
                        for step, event in chunk.items():
                            if not event or not (step in ["model", "tools"]):
                                continue
                            messages = event.get("messages", [])
                            if len(messages) == 0:
                                continue
                            message = messages[0]

                            # Tool 제한에 도달한 후에는 스트림만 소비하고 이벤트를 전송하지 않음
                            # (LangGraph 상태를 정상 유지하기 위해 스트림을 끝까지 소비해야 함)
                            if tool_limit_reached:
                                continue

                            if step == "model":
                                tool_calls = message.tool_calls
                                if not tool_calls:
                                    continue
                                tool = tool_calls[0]
                                if tool.get("name") == "ChatResponse":
                                    args = tool.get("args", {})
                                    metadata = args.get("metadata")
                                    custom_logger.info("========================================")
                                    custom_logger.info(args)
                                    yield f'{{"step": "done", "message_id": {json.dumps(args.get("message_id"))}, "role": "assistant", "content": {json.dumps(args.get("content"), ensure_ascii=False)}, "metadata": {json.dumps(self._handle_metadata(metadata), ensure_ascii=False)}, "created_at": "{datetime.utcnow().isoformat()}"}}'
                                else:
                                    yield f'{{"step": "model", "tool_calls": {json.dumps([tool["name"] for tool in tool_calls])}}}'
                            if step == "tools":
                                # ChatResponse는 Tool 호출 횟수에 포함하지 않음
                                if message.name != "ChatResponse":
                                    tool_call_count += 1
                                    custom_logger.info(f"Tool 호출 횟수: {tool_call_count}/{MAX_TOOL_CALLS}")
                                    if tool_call_count >= MAX_TOOL_CALLS:
                                        custom_logger.warning(f"Tool 호출 횟수 초과 ({MAX_TOOL_CALLS}회). 강제 응답 생성.")
                                        yield f'{{"step": "done", "message_id": "{uuid.uuid4()}", "role": "assistant", "content": "조회 가능한 데이터를 모두 확인했으나, 요청하신 조건에 해당하는 거래 데이터가 없습니다. 다른 지역이나 조건으로 다시 질문해주세요.", "metadata": {{}}, "created_at": "{datetime.utcnow().isoformat()}"}}'
                                        tool_limit_reached = True
                                        continue
                                yield f'{{"step": "tools", "name": {json.dumps(message.name)}, "content": {json.dumps(message.content, ensure_ascii=False)}}}'
                    except Exception as e:
                        # 청크 처리 중 예외 발생
                        custom_logger.error(f"Error processing chunk: {e}")
                        import traceback
                        custom_logger.error(traceback.format_exc())
                        error_response = {
                            "step": "done",
                            "message_id": str(uuid.uuid4()),
                            "role": "assistant",
                            "content": "데이터 처리 중 오류가 발생했습니다.",
                            "metadata": {},
                            "created_at": datetime.utcnow().isoformat(),
                            "error": str(e)
                        }
                        yield json.dumps(error_response, ensure_ascii=False)
                        break

                    agent_task = asyncio.create_task(agent_iterator.__anext__())

            if progress_task is not None:
                progress_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await progress_task

            while not self.progress_queue.empty():
                try:
                    remaining = self.progress_queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                yield json.dumps(remaining, ensure_ascii=False)

        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            custom_logger.error(f"Error in process_query: {e}")
            custom_logger.error(error_trace)
            
            error_content = f"처리 중 오류가 발생했습니다. 다시 시도해주세요."
            error_metadata = {}
            
            # 에러 응답을 스트리밍으로 전송 (HTTPException 대신)
            error_response = {
                "step": "done",
                "message_id": str(uuid.uuid4()),
                "role": "assistant",
                "content": error_content,
                "metadata": error_metadata,
                "created_at": datetime.utcnow().isoformat(),
                "error": str(e) if not isinstance(e, GraphRecursionError) else None
            }
            yield json.dumps(error_response, ensure_ascii=False)

    @log_execution
    def _handle_metadata(self, metadata) -> dict:
        custom_logger.info("========================================")
        custom_logger.info(metadata)
        result = {}
        if metadata:
            for k, v in metadata.items():
                result[k] = v
        return result
