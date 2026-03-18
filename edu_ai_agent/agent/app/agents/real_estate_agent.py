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

from app.agents.prompts import system_prompt
from app.agents.tools import search_apartment_trades, search_apartment_rentals
from app.core.config import settings


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
        tools=[search_apartment_trades, search_apartment_rentals],
        system_prompt=system_prompt,
        response_format=ToolStrategy(ChatResponse),
        checkpointer=checkpointer,
    )

    return agent
