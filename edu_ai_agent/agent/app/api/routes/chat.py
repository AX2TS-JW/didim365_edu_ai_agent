import json
import uuid
from datetime import datetime

from app.utils.logger import custom_logger
from app.utils.prompt_guard import detect_injection, check_leakage, REJECTION_MESSAGE
from fastapi import APIRouter, HTTPException
from app.models.chat import ChatRequest
from app.services.agent_service import AgentService
from app.services.graph_agent_service import GraphAgentService
from fastapi.responses import StreamingResponse

chat_router = APIRouter()

# 에이전트 모드: "deep" = Deep Agent (4주차), "react" = ReAct (기본), "graph" = StateGraph 전체
import os
_AGENT_MODE = os.getenv("AGENT_MODE", "deep")

if _AGENT_MODE == "graph":
    _agent_service = GraphAgentService()
else:
    # react와 deep 모두 AgentService 사용 (내부에서 AGENT_MODE로 분기)
    _agent_service = AgentService()


def _make_done_event(content: str) -> str:
    """step=done SSE 이벤트를 생성합니다."""
    response = {
        "step": "done",
        "message_id": str(uuid.uuid4()),
        "role": "assistant",
        "content": content,
        "metadata": {},
        "created_at": datetime.utcnow().isoformat(),
    }
    return f"data: {json.dumps(response, ensure_ascii=False)}\n\n"


@chat_router.post("/chat")
async def post_chat(request: ChatRequest):
    """
    자연어 쿼리를 에이전트가 처리합니다.

    ## 실제 테스트용 Request json
    ```json
    {
        "thread_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
        "message": "안녕하세요, 오늘 날씨가 어때요?"
    }
    ```
    """
    custom_logger.info(f"API Request: {request}")
    try:
        thread_id = getattr(request, "thread_id", uuid.uuid4())

        # ── B. 입력 필터링: 프롬프트 인젝션 탐지 ──
        if detect_injection(request.message):
            async def rejection_generator():
                yield _make_done_event(REJECTION_MESSAGE)

            return StreamingResponse(
                rejection_generator(),
                media_type="text/event-stream"
            )

        async def event_generator():
            try:
                yield f'data: {{"step": "model", "tool_calls": ["Planning"]}}\n\n'
                async for chunk in _agent_service.process_query(
                    user_messages=request.message,
                    thread_id=thread_id
                ):
                    # ── C. 출력 필터링: 시스템 프롬프트 유출 검사 ──
                    try:
                        parsed = json.loads(chunk)
                        step = parsed.get("step", "")
                        content = parsed.get("content", "")
                        if step == "done" and check_leakage(content, step):
                            yield _make_done_event(REJECTION_MESSAGE)
                            return
                    except (json.JSONDecodeError, AttributeError):
                        pass  # JSON 파싱 실패 시 원본 그대로 전달

                    yield f"data: {chunk}\n\n"
            except Exception as e:
                # 스트리밍 중 예외 발생 시 에러 메시지를 스트리밍으로 전송
                error_response = {
                    "step": "done",
                    "message_id": str(uuid.uuid4()),
                    "role": "assistant",
                    "content": "요청 처리 중 오류가 발생했습니다. 다시 시도해주세요.",
                    "metadata": {},
                    "created_at": datetime.utcnow().isoformat(),
                    "error": str(e)
                }
                yield f"data: {json.dumps(error_response, ensure_ascii=False)}\n\n"
                custom_logger.error(f"Error in event_generator: {e}", exc_info=True)
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream"
        )
        
    except Exception as e:
        # 스트리밍 시작 전 예외만 HTTPException으로 처리
        custom_logger.error(f"Error in /chat (before streaming): {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

