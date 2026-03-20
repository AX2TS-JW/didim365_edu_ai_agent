"""
캐시 TTL 만료 로직 테스트

tools.py의 _is_cache_expired() 함수를 검증합니다:
- 현재 월 데이터: 6시간 TTL
- 과거 월 데이터: 7일 TTL
- fetched_at 없는 경우: 만료 처리
"""

from datetime import datetime, timezone, timedelta
from unittest.mock import patch

import pytest

from app.agents.tools import _is_cache_expired


def _make_hits(fetched_at: str | None, count: int = 1) -> list:
    """테스트용 ES hit 목록을 생성합니다."""
    return [{"fetched_at": fetched_at, "apt_name": f"test_{i}"} for i in range(count)]


class TestIsCacheExpired:
    """_is_cache_expired 함수 테스트"""

    # ── 현재 월 (6시간 TTL) ──

    def test_current_month_fresh_cache(self):
        """현재 월 데이터: 1시간 전 캐시 → 유효"""
        now = datetime.now(timezone.utc)
        fresh = (now - timedelta(hours=1)).isoformat()
        current_ym = now.strftime("%Y%m")

        assert _is_cache_expired(_make_hits(fresh), current_ym) is False

    def test_current_month_expired_cache(self):
        """현재 월 데이터: 7시간 전 캐시 → 만료"""
        now = datetime.now(timezone.utc)
        old = (now - timedelta(hours=7)).isoformat()
        current_ym = now.strftime("%Y%m")

        assert _is_cache_expired(_make_hits(old), current_ym) is True

    def test_current_month_boundary(self):
        """현재 월 데이터: 정확히 6시간 경계 → 만료"""
        now = datetime.now(timezone.utc)
        boundary = (now - timedelta(hours=6, seconds=1)).isoformat()
        current_ym = now.strftime("%Y%m")

        assert _is_cache_expired(_make_hits(boundary), current_ym) is True

    # ── 과거 월 (7일 TTL) ──

    def test_past_month_fresh_cache(self):
        """과거 월 데이터: 1일 전 캐시 → 유효"""
        now = datetime.now(timezone.utc)
        fresh = (now - timedelta(days=1)).isoformat()

        assert _is_cache_expired(_make_hits(fresh), "202501") is False

    def test_past_month_expired_cache(self):
        """과거 월 데이터: 8일 전 캐시 → 만료"""
        now = datetime.now(timezone.utc)
        old = (now - timedelta(days=8)).isoformat()

        assert _is_cache_expired(_make_hits(old), "202501") is True

    def test_past_month_within_ttl(self):
        """과거 월 데이터: 6일 전 캐시 → 유효"""
        now = datetime.now(timezone.utc)
        within = (now - timedelta(days=6)).isoformat()

        assert _is_cache_expired(_make_hits(within), "202501") is False

    # ── 엣지 케이스 ──

    def test_no_fetched_at(self):
        """fetched_at 필드 없는 데이터 → 만료 처리"""
        assert _is_cache_expired(_make_hits(None), "202501") is True

    def test_empty_hits(self):
        """빈 리스트 → 만료 처리"""
        assert _is_cache_expired([], "202501") is True

    def test_mixed_fetched_at(self):
        """여러 건 중 가장 최근 fetched_at 기준으로 판단"""
        now = datetime.now(timezone.utc)
        old = (now - timedelta(days=10)).isoformat()
        recent = (now - timedelta(hours=1)).isoformat()

        hits = [
            {"fetched_at": old, "apt_name": "old"},
            {"fetched_at": recent, "apt_name": "recent"},
        ]
        # 가장 최근이 1시간 전이므로 과거 월 7일 TTL 기준 유효
        assert _is_cache_expired(hits, "202501") is False

    def test_z_suffix_utc_format(self):
        """fetched_at이 'Z' 접미사 UTC 형식일 때도 정상 파싱"""
        now = datetime.now(timezone.utc)
        fresh = (now - timedelta(hours=1)).strftime("%Y-%m-%dT%H:%M:%S.%fZ")

        assert _is_cache_expired(_make_hits(fresh), "202501") is False
