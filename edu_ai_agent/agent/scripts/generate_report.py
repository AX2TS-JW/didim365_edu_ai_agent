"""
2주차 5일차: 성능 진단 리포트 생성 스크립트 (Mini Project 2)

Day 2~4 결과를 집계하여 종합 성능 진단 리포트를 생성합니다.
- diagnostic_report.json (Day 2: 진단 케이스)
- judge_report.json (Day 3: LLM 판사 채점)
- deepeval_report.json (Day 4: DeepEval + ToolUsage 메트릭)

사용법:
  cd edu_ai_agent/agent
  uv run python scripts/generate_report.py
"""

import json
from collections import Counter
from datetime import datetime
from pathlib import Path

OUTPUT_DIR = Path(__file__).parent / "output"
REPORT_FILE = Path(__file__).parent.parent / "docs" / "performance_report.md"
SUMMARY_FILE = OUTPUT_DIR / "report_summary.json"


def load_json(path: Path) -> dict | list | None:
    """JSON 파일을 로드합니다. 없으면 None 반환."""
    if not path.exists():
        print(f"  ⚠️  파일 없음: {path.name}")
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def main():
    print("=== 성능 진단 리포트 생성 ===\n")

    # 데이터 로드
    diagnostic = load_json(OUTPUT_DIR / "diagnostic_report.json")
    judge = load_json(OUTPUT_DIR / "judge_report.json")
    deepeval = load_json(OUTPUT_DIR / "deepeval_report.json")

    if not any([diagnostic, judge, deepeval]):
        print("❌ 분석할 데이터가 없습니다. Day 2~4 스크립트를 먼저 실행하세요.")
        return

    # 리포트 생성
    sections = []
    summary_data = {"generated_at": datetime.now().isoformat()}

    # ── 1. Executive Summary ──
    sections.append("# 부동산 실거래가 에이전트 성능 진단 리포트\n")
    sections.append(f"> 생성일: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")

    # ── 2. 진단 케이스 분석 (Day 2) ──
    if diagnostic:
        sections.append("## 1. 진단 케이스 분석 (에이전트 취약점 탐색)\n")
        total = len(diagnostic)
        passed = sum(1 for d in diagnostic if d.get("diagnosis", {}).get("pass"))
        sections.append(f"- 총 {total}개 진단 케이스 중 **{passed}개 통과** ({passed/total*100:.0f}%)\n")

        # 카테고리별 결과
        sections.append("| 카테고리 | 케이스 | 결과 | 요약 |")
        sections.append("|----------|--------|------|------|")
        for d in diagnostic:
            diag = d.get("diagnosis", {})
            status = "✅" if diag.get("pass") else "❌"
            sections.append(
                f"| {d.get('category', '')} | {d.get('input', '')[:30]} | "
                f"{status} | {diag.get('summary', '')} |"
            )
        sections.append("")

        # 실패 패턴 분석
        failures = [d for d in diagnostic if not d.get("diagnosis", {}).get("pass")]
        if failures:
            sections.append("### 실패 패턴\n")
            fail_categories = Counter(d.get("category", "") for d in failures)
            for cat, count in fail_categories.most_common():
                sections.append(f"- **{cat}**: {count}건")
            sections.append("")

        summary_data["diagnostic"] = {
            "total": total,
            "passed": passed,
            "fail_categories": dict(Counter(
                d.get("category", "") for d in failures
            )) if failures else {},
        }

    # ── 3. LLM 판사 채점 결과 (Day 3) ──
    if judge:
        sections.append("## 2. LLM-as-a-Judge 채점 결과\n")
        summary = judge.get("summary", {})
        avg = summary.get("average_score", 0)
        sections.append(f"- 평균 점수: **{avg:.2f} / 5.00**")
        sections.append(f"- 판사 모델: {summary.get('judge_model', 'N/A')}")
        sections.append(f"- 평가 건수: {summary.get('total_cases', 0)}건\n")

        sections.append("| 질문 | 점수 | 근거 |")
        sections.append("|------|------|------|")
        for r in judge.get("results", []):
            score = r.get("judge_score", 0)
            emoji = "🟢" if score >= 4 else "🟡" if score >= 3 else "🔴"
            sections.append(
                f"| {r.get('input', '')[:35]} | {emoji} {score}/5 | "
                f"{r.get('judge_reasoning', '')[:60]} |"
            )
        sections.append("")

        # 점수대별 분포
        score_dist = Counter(r.get("judge_score", 0) for r in judge.get("results", []))
        sections.append("### 점수 분포\n")
        for s in range(5, 0, -1):
            bar = "█" * score_dist.get(s, 0)
            sections.append(f"- {s}점: {bar} ({score_dist.get(s, 0)}건)")
        sections.append("")

        summary_data["judge"] = {
            "average_score": avg,
            "score_distribution": dict(score_dist),
        }

    # ── 4. DeepEval 메트릭 결과 (Day 4) ──
    if deepeval:
        sections.append("## 3. DeepEval 에이전틱 메트릭 결과\n")
        de_summary = deepeval.get("summary", {})
        sections.append(f"- 평가 모델: {de_summary.get('eval_model', 'N/A')}")
        sections.append(f"- Tool 사용 평균 점수: **{de_summary.get('tool_usage_avg_score', 0):.3f}**\n")

        avg_scores = de_summary.get("deepeval_averages", {})
        if avg_scores:
            sections.append("### DeepEval 메트릭 평균\n")
            sections.append("| 메트릭 | 평균 점수 |")
            sections.append("|--------|----------|")
            for name, score in avg_scores.items():
                sections.append(f"| {name} | {score if score is not None else 'N/A'} |")
            sections.append("")

        # Tool 사용 상세
        sections.append("### Tool 사용 메트릭 상세\n")
        sections.append("| 질문 | Tool 점수 | 결과 |")
        sections.append("|------|----------|------|")
        for r in deepeval.get("results", []):
            tu = r.get("tool_usage", {})
            status = "✅" if tu.get("pass") else "❌"
            sections.append(
                f"| {r.get('input', '')[:30]} | {tu.get('score', 0)} | "
                f"{status} {tu.get('summary', '')} |"
            )
        sections.append("")

        summary_data["deepeval"] = {
            "averages": avg_scores,
            "tool_usage_avg": de_summary.get("tool_usage_avg_score", 0),
        }

    # ── 5. 개선 우선순위 ──
    sections.append("## 4. 개선 우선순위\n")
    priorities = _derive_priorities(diagnostic, judge, deepeval)
    for i, p in enumerate(priorities, 1):
        sections.append(f"{i}. **{p['title']}** (영향도: {p['impact']})")
        sections.append(f"   - {p['description']}")
        sections.append(f"   - 권장 조치: {p['action']}")
    sections.append("")

    summary_data["priorities"] = priorities

    # ── 저장 ──
    report_content = "\n".join(sections)
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_FILE, "w", encoding="utf-8") as f:
        f.write(report_content)
    print(f"📄 리포트 저장: {REPORT_FILE}")

    with open(SUMMARY_FILE, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, ensure_ascii=False, indent=2, default=str)
    print(f"📊 요약 JSON: {SUMMARY_FILE}")


def _derive_priorities(diagnostic, judge, deepeval) -> list[dict]:
    """분석 결과에서 개선 우선순위를 도출합니다."""
    priorities = []

    # 진단 케이스 기반
    if diagnostic:
        failures = [d for d in diagnostic if not d.get("diagnosis", {}).get("pass")]
        fail_cats = Counter(d.get("category", "") for d in failures)
        for cat, count in fail_cats.most_common(3):
            priorities.append({
                "title": f"{cat} 처리 개선",
                "impact": "높음" if count >= 2 else "중간",
                "description": f"진단 테스트에서 {count}건 실패",
                "action": _suggest_action(cat),
            })

    # 판사 점수 기반
    if judge:
        low_scores = [
            r for r in judge.get("results", [])
            if r.get("judge_score", 5) <= 2
        ]
        if low_scores:
            priorities.append({
                "title": "저점수 답변 품질 개선",
                "impact": "높음",
                "description": f"1~2점 답변 {len(low_scores)}건 — 근거: {low_scores[0].get('judge_reasoning', '')[:50]}",
                "action": "시스템 프롬프트에 답변 구조화 지침 강화 및 누락 데이터 처리 로직 보완",
            })

    # DeepEval 기반
    if deepeval:
        avgs = deepeval.get("summary", {}).get("deepeval_averages", {})
        for metric_name, score in avgs.items():
            if score is not None and score < 0.7:
                priorities.append({
                    "title": f"{metric_name} 개선",
                    "impact": "중간",
                    "description": f"평균 {score:.3f} (기준 0.7 미달)",
                    "action": "프롬프트 또는 도구 출력 포맷 조정",
                })

    if not priorities:
        priorities.append({
            "title": "전반적으로 양호",
            "impact": "낮음",
            "description": "주요 실패 패턴 없음",
            "action": "엣지 케이스 테스트 추가 및 데이터셋 확대",
        })

    return priorities


def _suggest_action(category: str) -> str:
    """카테고리별 개선 조치를 제안합니다."""
    actions = {
        "모호한 지역명": "지역코드 매핑 로직에 퍼지 매칭 또는 사용자 확인 로직 추가",
        "미래 월 조회": "시스템 프롬프트의 미래 월 차단 규칙 강화",
        "복합 질의 (2개 도구)": "복합 질문 분해 로직 또는 프롬프트 지침 추가",
        "동 단위 → 시군구 매핑": "동→시군구 매핑 데이터 추가 (ES 또는 폴백 딕셔너리)",
        "필터 조건 포함 (동+평수)": "프롬프트에 필터링 가이드 추가 (동, 면적 범위)",
        "지원 안 되는 지역": "폴백 딕셔너리 확대 또는 ES 지역코드 인덱스 보강",
        "프롬프트 인젝션": "프롬프트 인젝션 방어 규칙 점검",
        "지역 누락": "프롬프트에 필수 정보 확인 로직 강화",
        "출력 형식 지정": "프롬프트에 출력 형식 지원 가이드 추가",
        "시점 비교 (2개 년월)": "시점 비교 질문에 대한 날짜 계산 로직 점검",
    }
    return actions.get(category, "해당 카테고리에 대한 추가 분석 필요")


if __name__ == "__main__":
    main()
