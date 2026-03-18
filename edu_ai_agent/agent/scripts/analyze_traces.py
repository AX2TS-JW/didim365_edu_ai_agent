"""
2주차 2일차: Opik 트레이스 분석 스크립트

Opik 대시보드에 기록된 에이전트 실행 로그를 프로그래밍 방식으로 조회하여
각 추론 단계(질문 → 도구 선택 → 도구 결과 → 최종 답변)를 구조화 출력합니다.
"""

import os
from dotenv import load_dotenv

load_dotenv()

os.environ["OPIK_URL_OVERRIDE"] = os.getenv("OPIK__URL_OVERRIDE", "")
os.environ["OPIK_WORKSPACE"] = os.getenv("OPIK__WORKSPACE", "default")

from opik import Opik

PROJECT_NAME = os.getenv("OPIK__PROJECT", "jae-project")

client = Opik(
    host=os.getenv("OPIK__URL_OVERRIDE"),
    project_name=PROJECT_NAME,
)


def analyze_traces(max_traces: int = 20):
    """프로젝트의 최근 트레이스를 조회하고 추론 단계를 분석합니다."""
    traces = client.get_traces(project_name=PROJECT_NAME)

    if not traces:
        print("트레이스가 없습니다. 에이전트에 몇 가지 질문을 먼저 해보세요.")
        return

    print(f"=== {PROJECT_NAME} 트레이스 분석 ({len(traces)}건) ===\n")

    for i, trace in enumerate(traces[:max_traces]):
        print(f"{'─' * 70}")
        print(f"[Trace {i + 1}] ID: {trace.id}")
        print(f"  입력: {trace.input}")
        print(f"  출력: {_truncate(str(trace.output), 200)}")

        # 시간 정보
        if trace.start_time and trace.end_time:
            duration = (trace.end_time - trace.start_time).total_seconds()
            print(f"  소요시간: {duration:.1f}초")

        # 토큰 사용량
        usage = trace.usage
        if usage:
            print(f"  토큰: 입력={usage.get('prompt_tokens', '?')}, "
                  f"출력={usage.get('completion_tokens', '?')}, "
                  f"합계={usage.get('total_tokens', '?')}")

        # 에러 여부
        if trace.error_info:
            print(f"  ❌ 에러: {trace.error_info}")

        # 하위 스팬(도구 호출 등) 분석
        try:
            spans = client.get_spans(trace_id=trace.id)
            if spans:
                print(f"  추론 단계 ({len(spans)}개 스팬):")
                for j, span in enumerate(spans):
                    span_type = span.type or "unknown"
                    span_name = span.name or "unnamed"
                    span_duration = ""
                    if span.start_time and span.end_time:
                        sd = (span.end_time - span.start_time).total_seconds()
                        span_duration = f" ({sd:.1f}s)"

                    prefix = "    → " if j > 0 else "    ● "
                    print(f"{prefix}[{span_type}] {span_name}{span_duration}")

                    # 도구 호출이면 입출력 요약
                    if span_type == "tool" or "tool" in span_name.lower():
                        if span.input:
                            print(f"      입력: {_truncate(str(span.input), 150)}")
                        if span.output:
                            print(f"      출력: {_truncate(str(span.output), 150)}")
        except Exception as e:
            print(f"  (스팬 조회 실패: {e})")

        print()

    # 요약 통계
    print(f"{'═' * 70}")
    print(f"총 {len(traces)}건의 트레이스")
    error_count = sum(1 for t in traces if t.error_info)
    if error_count:
        print(f"⚠️  에러 발생: {error_count}건")


def _truncate(text: str, max_len: int = 200) -> str:
    """긴 텍스트를 잘라서 반환합니다."""
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


if __name__ == "__main__":
    analyze_traces()
