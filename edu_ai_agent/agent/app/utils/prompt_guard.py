"""
프롬프트 인젝션 방어 모듈

A. 시스템 프롬프트 강화 → prompts.py에 직접 적용
B. 입력 필터링 (정규식) → detect_injection()
C. 출력 필터링 (유출 검사) → check_leakage()
"""

import re

from app.utils.logger import custom_logger

# ── B. 입력 필터링: 프롬프트 인젝션 탐지 패턴 ──

_INJECTION_PATTERNS = [
    # 시스템 프롬프트 추출 시도
    r"시스템\s*프롬프트",
    r"system\s*prompt",
    r"지시사항.*보여",
    r"지시사항.*알려",
    r"너의\s*(설정|역할|규칙|도구|프롬프트)",
    r"내부\s*(설정|규칙|지시)",

    # 역할 변경 / 탈옥 시도
    r"ignore\s*(previous|above|all)\s*instructions",
    r"disregard\s*(previous|above|all)\s*instructions",
    r"pretend\s*you\s*are",
    r"act\s*as\s*if",
    r"you\s*are\s*now",
    r"새로운\s*역할",
    r"역할을?\s*바꿔",
    r"지시.*무시",
    r"규칙.*무시",

    # 구조화된 출력을 통한 정보 추출
    r"(JSON|xml|yaml).*출력.*설정",
    r"(JSON|xml|yaml).*형식.*규칙",
    r"디버깅.*출력",
    r"설정.*dump",
    r"config.*출력",

    # 간접 추출
    r"어떤.*요청.*거절",
    r"하면\s*안\s*되는\s*질문",
    r"답변.*규칙.*알려",
]

_COMPILED_PATTERNS = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]

REJECTION_MESSAGE = (
    "죄송합니다. 해당 요청은 처리할 수 없습니다. "
    "부동산 실거래가 관련 질문을 해주세요!"
)


def detect_injection(user_input: str) -> bool:
    """
    사용자 입력에서 프롬프트 인젝션 패턴을 탐지합니다.

    Returns:
        True이면 인젝션 의심, False이면 안전
    """
    for pattern in _COMPILED_PATTERNS:
        if pattern.search(user_input):
            custom_logger.warning(
                f"프롬프트 인젝션 탐지: pattern='{pattern.pattern}', "
                f"input='{user_input[:100]}...'"
            )
            return True
    return False


# ── C. 출력 필터링: 시스템 프롬프트 유출 검사 ──

# 시스템 프롬프트에만 존재하는 핵심 문구들 (유출 시 탐지)
_SENSITIVE_PHRASES = [
    "부동산 실거래가 분석 전문 AI 어시스턴트",
    "search_apartment_trades",
    "search_apartment_rentals",
    "Tool 호출은 최대 5회",
    "Tool 선택 기준",
    "cdealType",
    "ChatResponse",
    "Response Format",
    "message_id (필수 키값)",
    "프롬프트 인젝션",
    "인젝션 시도",
]

# 정상 응답에서 등장할 수 있는 문구는 제외 (tool 이름은 도구 실행 결과에 나올 수 있음)
_WHITELIST_CONTEXTS = [
    "search_apartment_trades",  # 도구 실행 결과 step에서는 허용
    "search_apartment_rentals",
]


def check_leakage(response_content: str, step: str = "done") -> bool:
    """
    LLM 응답에서 시스템 프롬프트 핵심 문구의 유출을 검사합니다.

    Args:
        response_content: LLM 응답 텍스트
        step: 현재 스트리밍 단계 ("model", "tools", "done")

    Returns:
        True이면 유출 감지, False이면 안전
    """
    if not response_content:
        return False

    content_lower = response_content.lower()

    for phrase in _SENSITIVE_PHRASES:
        phrase_lower = phrase.lower()
        if phrase_lower in content_lower:
            # tools 단계에서는 도구 이름이 자연스럽게 등장할 수 있으므로 허용
            if step == "tools" and phrase in _WHITELIST_CONTEXTS:
                continue

            custom_logger.warning(
                f"시스템 프롬프트 유출 감지: phrase='{phrase}', "
                f"step='{step}', content='{response_content[:100]}...'"
            )
            return True

    return False
