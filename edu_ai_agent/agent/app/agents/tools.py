"""
부동산 실거래가 조회 Tool 정의.
- ES 캐시 우선 조회 → 없으면 data.go.kr API 호출 → ES에 저장
- 아파트 매매 실거래가 상세 API (데이터셋 15126468)
- 아파트 전월세 실거래가 API (데이터셋 15126474)

[LangChain 매핑]
- https://docs.langchain.com/oss/python/langchain/quickstart → Step 3: Tool Creation
- https://docs.langchain.com/oss/python/langchain/agents → Tools Management > Static Tools
"""

from datetime import datetime, timezone, timedelta
from dateutil.relativedelta import relativedelta

import httpx
import xmltodict
from langchain.tools import tool

from app.core.config import settings
from app.core.es_client import es_client
from app.core.es_index_setup import IDX_APT_TRADES, IDX_APT_RENTALS, IDX_REGION_CODES
from app.utils.logger import custom_logger

# ES 미연결 시 폴백용 지역코드 매핑
_FALLBACK_REGION_CODES = {
    "강남구": "11680", "서초구": "11650", "송파구": "11710",
    "강동구": "11740", "마포구": "11440", "용산구": "11170",
    "성동구": "11200", "광진구": "11215", "동작구": "11590",
    "영등포구": "11560", "강서구": "11500", "양천구": "11470",
    "구로구": "11530", "금천구": "11545", "관악구": "11620",
    "노원구": "11350", "도봉구": "11320", "강북구": "11305",
    "성북구": "11290", "종로구": "11110", "중구": "11140",
    "중랑구": "11260", "동대문구": "11230", "은평구": "11380",
    "서대문구": "11410",
    "분당구": "41135", "수원시": "41110", "고양시": "41280",
    "성남시": "41130", "용인시": "41460", "화성시": "41590",
    "해운대구": "26350", "수성구": "27260", "연수구": "28185",
    "유성구": "30200", "광산구": "29200", "세종시": "36110",
}


# ──────────────────────────────────────────────────────────
# 지역코드 조회
# ──────────────────────────────────────────────────────────

def _resolve_region_code(region: str) -> tuple[str | None, str | None]:
    """ES에서 지역코드를 검색합니다. ES 미연결 시 폴백 딕셔너리 사용.

    Returns:
        (region_code, error_hint): 성공 시 (코드, None), 실패 시 (None, 안내 메시지)
    """
    if es_client is not None:
        try:
            result = es_client.search(
                index=IDX_REGION_CODES,
                body={
                    "query": {
                        "bool": {
                            "should": [
                                {"term": {"region_name": region}},
                                {"match": {"aliases": region}},
                            ]
                        }
                    },
                    "size": 1,
                },
            )
            hits = result["hits"]["hits"]
            if hits:
                return hits[0]["_source"]["region_code"], None
        except Exception as e:
            custom_logger.error(f"ES 지역코드 검색 실패: {e}")

    # 폴백: 하드코딩 딕셔너리
    code = _resolve_region_code_fallback(region)
    if code:
        return code, None

    # 동/읍/면/리 이름 감지 → 시군구 단위 안내
    if region.endswith(("동", "읍", "면", "리")):
        return None, (f"[조회 불가] '{region}'은(는) 동/읍/면 이름입니다. "
                      f"실거래가 API는 기초자치단체(구/시/군) 단위로만 조회 가능합니다. "
                      f"이 메시지를 사용자에게 전달하고, 다른 지역으로 재조회하지 마세요. "
                      f"사용자에게 기초자치단체를 되물어보세요.")

    # 시/도(광역자치단체) 감지 → 기초자치단체 안내
    _SIDO_NAMES = ("서울", "부산", "대구", "인천", "광주", "대전", "울산", "세종",
                   "경기", "강원", "충북", "충남", "전북", "전남", "경북", "경남", "제주")
    region_stripped = region.replace("특별시", "").replace("광역시", "").replace("특별자치시", "").replace("특별자치도", "").replace("도", "").replace("시", "").replace(" ", "")
    if region_stripped in _SIDO_NAMES or region in _SIDO_NAMES:
        return None, (f"[조회 불가] '{region}'은(는) 광역자치단체(시/도)입니다. "
                      f"실거래가 API는 기초자치단체(구/시/군) 단위로만 조회 가능합니다. "
                      f"이 메시지를 사용자에게 전달하고, 다른 지역으로 재조회하지 마세요. "
                      f"사용자에게 '{region}' 내 어느 기초자치단체를 조회할지 되물어보세요.")

    return None, (f"[조회 불가] '{region}'에 해당하는 지역코드를 찾을 수 없습니다. "
                  f"기초자치단체(구/시/군) 단위로 입력해주세요. (예: 강남구, 분당구) "
                  f"이 메시지를 사용자에게 전달하고, 다른 지역으로 재조회하지 마세요.")


def _resolve_region_code_fallback(region: str) -> str | None:
    """폴백 딕셔너리에서 지역코드를 찾습니다."""
    if region in _FALLBACK_REGION_CODES:
        return _FALLBACK_REGION_CODES[region]
    for name, code in _FALLBACK_REGION_CODES.items():
        if region in name or name in region:
            return code
    return None


# ──────────────────────────────────────────────────────────
# 금액 파싱
# ──────────────────────────────────────────────────────────

def _parse_price(price_str: str) -> float | str:
    """금액 문자열을 만원 단위 숫자로 변환합니다."""
    try:
        return float(price_str.replace(",", ""))
    except (ValueError, AttributeError):
        return price_str


# ──────────────────────────────────────────────────────────
# ES 캐시 조회 / 저장
# ──────────────────────────────────────────────────────────

# 캐시 TTL: 현재 월 데이터는 신고 지연으로 계속 추가되므로 짧게,
# 과거 월 데이터는 변동이 적으므로 길게 설정
_CACHE_TTL_CURRENT_MONTH = timedelta(hours=6)
_CACHE_TTL_PAST_MONTH = timedelta(days=7)


def _is_cache_expired(hits: list, year_month: str) -> bool:
    """캐시 데이터의 fetched_at을 확인하여 TTL 초과 여부를 판단합니다."""
    current_ym = datetime.now().strftime("%Y%m")
    ttl = _CACHE_TTL_CURRENT_MONTH if year_month >= current_ym else _CACHE_TTL_PAST_MONTH

    # 가장 최근 fetched_at 기준으로 판단
    latest_fetched = None
    for h in hits:
        fetched_at = h.get("fetched_at")
        if not fetched_at:
            continue
        try:
            dt = datetime.fromisoformat(fetched_at.replace("Z", "+00:00"))
            if latest_fetched is None or dt > latest_fetched:
                latest_fetched = dt
        except (ValueError, AttributeError):
            continue

    if latest_fetched is None:
        return True  # fetched_at 없으면 만료 처리

    age = datetime.now(timezone.utc) - latest_fetched
    if age > ttl:
        custom_logger.info(
            f"캐시 TTL 초과: year_month={year_month}, age={age}, ttl={ttl}"
        )
        return True
    return False


def _search_trades_from_es(region_code: str, year_month: str) -> list | None:
    """ES에서 매매 거래 데이터를 조회합니다. TTL 초과 시 None 반환(API 재호출 유도)."""
    if es_client is None:
        return None
    try:
        result = es_client.search(
            index=IDX_APT_TRADES,
            body={
                "query": {"bool": {"filter": [
                    {"term": {"region_code": region_code}},
                    {"term": {"year_month": year_month}},
                ]}},
                "size": 100,
                "sort": [{"deal_amount": {"order": "desc"}}],
            },
        )
        hits = result["hits"]["hits"]
        if not hits:
            return None
        sources = [h["_source"] for h in hits]
        if _is_cache_expired(sources, year_month):
            return None  # TTL 초과 → 캐시 미스 처리 → API 재호출
        return sources
    except Exception as e:
        custom_logger.error(f"ES 매매 데이터 조회 실패: {e}")
        return None


def _save_trades_to_es(items: list, region_code: str, year_month: str):
    """API 응답 데이터를 ES에 벌크 저장합니다."""
    if es_client is None or not items:
        return
    now = datetime.now(timezone.utc).isoformat()
    actions = []
    for item in items:
        doc = {
            "region_code": region_code,
            "year_month": year_month,
            "apt_name": (item.get("aptNm") or "").strip(),
            "deal_amount": int(float((item.get("dealAmount") or "0").strip().replace(",", "") or 0)),
            "exclu_use_ar": float((item.get("excluUseAr") or "0").strip() or 0),
            "floor": (item.get("floor") or "").strip(),
            "umd_name": (item.get("umdNm") or "").strip(),
            "deal_day": (item.get("dealDay") or "").strip(),
            "dealing_gbn": (item.get("dealingGbn") or "").strip(),
            "cdeal_type": (item.get("cdealType") or "").strip(),
            "fetched_at": now,
        }
        # 고유 ID로 중복 방지
        doc_id = f"{region_code}_{year_month}_{doc['apt_name']}_{doc['floor']}_{doc['deal_day']}_{doc['deal_amount']}"
        actions.append({"index": {"_index": IDX_APT_TRADES, "_id": doc_id}})
        actions.append(doc)
    try:
        es_client.bulk(body=actions, refresh="wait_for")
        custom_logger.info(f"ES 저장: trades {region_code}/{year_month} ({len(items)}건)")
    except Exception as e:
        custom_logger.error(f"ES 벌크 저장 실패: {e}")


def _search_rentals_from_es(region_code: str, year_month: str) -> list | None:
    """ES에서 전월세 거래 데이터를 조회합니다. TTL 초과 시 None 반환(API 재호출 유도)."""
    if es_client is None:
        return None
    try:
        result = es_client.search(
            index=IDX_APT_RENTALS,
            body={
                "query": {"bool": {"filter": [
                    {"term": {"region_code": region_code}},
                    {"term": {"year_month": year_month}},
                ]}},
                "size": 100,
                "sort": [{"deposit": {"order": "desc"}}],
            },
        )
        hits = result["hits"]["hits"]
        if not hits:
            return None
        sources = [h["_source"] for h in hits]
        if _is_cache_expired(sources, year_month):
            return None  # TTL 초과 → 캐시 미스 처리 → API 재호출
        return sources
    except Exception as e:
        custom_logger.error(f"ES 전월세 데이터 조회 실패: {e}")
        return None


def _save_rentals_to_es(items: list, region_code: str, year_month: str):
    """전월세 API 응답 데이터를 ES에 벌크 저장합니다."""
    if es_client is None or not items:
        return
    now = datetime.now(timezone.utc).isoformat()
    actions = []
    for item in items:
        doc = {
            "region_code": region_code,
            "year_month": year_month,
            "apt_name": (item.get("aptNm") or "").strip(),
            "deposit": int(float((item.get("deposit") or "0").strip().replace(",", "") or 0)),
            "monthly_rent": int(float((item.get("monthlyRent") or "0").strip().replace(",", "") or 0)),
            "exclu_use_ar": float((item.get("excluUseAr") or "0").strip() or 0),
            "floor": (item.get("floor") or "").strip(),
            "umd_name": (item.get("umdNm") or "").strip(),
            "deal_day": (item.get("dealDay") or "").strip(),
            "fetched_at": now,
        }
        doc_id = f"{region_code}_{year_month}_{doc['apt_name']}_{doc['floor']}_{doc['deal_day']}_{doc['deposit']}"
        actions.append({"index": {"_index": IDX_APT_RENTALS, "_id": doc_id}})
        actions.append(doc)
    try:
        es_client.bulk(body=actions, refresh="wait_for")
        custom_logger.info(f"ES 저장: rentals {region_code}/{year_month} ({len(items)}건)")
    except Exception as e:
        custom_logger.error(f"ES 벌크 저장 실패: {e}")


# ──────────────────────────────────────────────────────────
# data.go.kr API 호출
# ──────────────────────────────────────────────────────────

def _fetch_trades_from_api(region_code: str, year_month: str) -> list | str:
    """data.go.kr 매매 실거래가 API를 호출합니다. 성공 시 item 리스트, 실패 시 에러 문자열."""
    api_key = settings.DATA_GO_KR_API_KEY
    if not api_key:
        return "data.go.kr API 키가 설정되지 않았습니다."

    url = "https://apis.data.go.kr/1613000/RTMSDataSvcAptTradeDev/getRTMSDataSvcAptTradeDev"
    params = {
        "serviceKey": api_key,
        "LAWD_CD": region_code,
        "DEAL_YMD": year_month,
        "pageNo": "1",
        "numOfRows": "100",
    }

    try:
        response = httpx.get(url, params=params, timeout=15.0)
        response.raise_for_status()
    except httpx.HTTPError as e:
        custom_logger.error(f"실거래가 API 호출 실패: {e}")
        return f"API 호출 중 오류가 발생했습니다: {e}"

    try:
        data = xmltodict.parse(response.text)
    except Exception as e:
        custom_logger.error(f"XML 파싱 실패: {e}")
        return f"응답 파싱 중 오류가 발생했습니다: {e}"

    body = data.get("response", {}).get("body", {})
    total_count = int(body.get("totalCount", 0))
    items = body.get("items", {})
    if not items or items == "":
        return [], 0

    item_list = items.get("item", [])
    if isinstance(item_list, dict):
        item_list = [item_list]
    return item_list, total_count


def _fetch_rentals_from_api(region_code: str, year_month: str) -> list | str:
    """data.go.kr 전월세 실거래가 API를 호출합니다."""
    api_key = settings.DATA_GO_KR_API_KEY
    if not api_key:
        return "data.go.kr API 키가 설정되지 않았습니다."

    url = "https://apis.data.go.kr/1613000/RTMSDataSvcAptRent/getRTMSDataSvcAptRent"
    params = {
        "serviceKey": api_key,
        "LAWD_CD": region_code,
        "DEAL_YMD": year_month,
        "pageNo": "1",
        "numOfRows": "100",
    }

    try:
        response = httpx.get(url, params=params, timeout=15.0)
        response.raise_for_status()
    except httpx.HTTPError as e:
        custom_logger.error(f"전월세 API 호출 실패: {e}")
        return f"API 호출 중 오류가 발생했습니다: {e}"

    try:
        data = xmltodict.parse(response.text)
    except Exception as e:
        custom_logger.error(f"XML 파싱 실패: {e}")
        return f"응답 파싱 중 오류가 발생했습니다: {e}"

    body = data.get("response", {}).get("body", {})
    total_count = int(body.get("totalCount", 0))
    items = body.get("items", {})
    if not items or items == "":
        return [], 0

    item_list = items.get("item", [])
    if isinstance(item_list, dict):
        item_list = [item_list]
    return item_list, total_count


# ──────────────────────────────────────────────────────────
# 포맷팅
# ──────────────────────────────────────────────────────────

def _format_trades(trades: list, region: str, region_code: str, year_month: str, from_es: bool = False, total_count: int = 0) -> str:
    """매매 거래 데이터를 사용자에게 보여줄 문자열로 포맷합니다."""
    # 해제 거래 필터링 (ES 데이터는 이미 cdeal_type 필드 사용)
    cdeal_key = "cdeal_type" if from_es else "cdealType"
    valid = [t for t in trades if not t.get(cdeal_key)]

    if not valid:
        return f"{region} {year_month} 기간에 유효한 거래 내역이 없습니다 (모두 해제됨)."

    def _format_one_trade(t, from_es):
        if from_es:
            apt_name = t.get("apt_name", "알수없음")
            price_val = t.get("deal_amount", 0)
            area = str(t.get("exclu_use_ar", "0"))
            floor = t.get("floor", "?")
            dong = t.get("umd_name", "")
            deal_day = t.get("deal_day", "?")
            trade_type = t.get("dealing_gbn", "")
        else:
            apt_name = (t.get("aptNm") or "알수없음").strip()
            price_val = _parse_price((t.get("dealAmount") or "0").strip().replace(",", ""))
            area = (t.get("excluUseAr") or "0").strip()
            floor = (t.get("floor") or "?").strip()
            dong = (t.get("umdNm") or "").strip()
            deal_day = (t.get("dealDay") or "?").strip()
            trade_type = (t.get("dealingGbn") or "").strip()

        price_display = f"{price_val / 10000:.1f}억원" if isinstance(price_val, (int, float)) else f"{price_val}만원"
        line = f"- {dong} {apt_name} | {area}㎡ | {floor}층 | {price_display} | {year_month[:4]}.{year_month[4:]}.{deal_day}"
        if trade_type:
            line += f" | {trade_type}"
        return line

    # 가격순 정렬 (통계 + 상위/하위 표시용)
    def _get_price(t):
        p = t.get("deal_amount", 0) if from_es else _parse_price((t.get("dealAmount") or "0").strip().replace(",", ""))
        return p if isinstance(p, (int, float)) else 0

    valid_sorted = sorted(valid, key=_get_price, reverse=True)
    total = len(valid_sorted)
    source = "ES 캐시" if from_es else "API"

    # 요약 통계 (전체 기준)
    prices = [_get_price(t) for t in valid_sorted if _get_price(t) > 0]
    stats = ""
    if prices:
        avg_p = sum(prices) / len(prices)
        max_p = max(prices)
        min_p = min(prices)
        stats = f"\n■ 요약 통계 ({total}건 전체 기준): 평균 {avg_p/10000:.1f}억원 | 최고 {max_p/10000:.1f}억원 | 최저 {min_p/10000:.1f}억원\n"

    # 상위 5건
    top_lines = [_format_one_trade(t, from_es) for t in valid_sorted[:5]]
    # 하위 5건 (상위와 겹치지 않도록)
    bottom_items = valid_sorted[-5:] if total > 10 else valid_sorted[5:]
    bottom_lines = [_format_one_trade(t, from_es) for t in reversed(bottom_items)] if bottom_items else []

    actual_total = total_count if total_count > 0 else total
    header = f"📊 {region}({region_code}) {year_month[:4]}년 {year_month[4:]}월 매매 실거래가 — 전체 {actual_total}건\n{stats}\n"

    body = "▸ 상위 매물\n" + "\n".join(top_lines)
    if bottom_lines:
        body += "\n\n▸ 하위 매물\n" + "\n".join(bottom_lines)

    footer = "\n\n⚠️ 참고: 실거래가 데이터는 신고 지연이 있어 최근 1~2개월은 데이터가 적을 수 있습니다."
    return header + body + footer


def _format_rentals(rentals: list, region: str, region_code: str, year_month: str, from_es: bool = False, total_count: int = 0) -> str:
    """전월세 거래 데이터를 사용자에게 보여줄 문자열로 포맷합니다."""
    if not rentals:
        return f"{region} {year_month} 기간에 전월세 거래 내역이 없습니다."

    def _format_one_rental(t, from_es):
        if from_es:
            apt_name = t.get("apt_name", "알수없음")
            deposit_val = t.get("deposit", 0)
            monthly_rent = str(t.get("monthly_rent", 0))
            area = str(t.get("exclu_use_ar", "0"))
            floor = t.get("floor", "?")
            dong = t.get("umd_name", "")
            deal_day = t.get("deal_day", "?")
        else:
            apt_name = (t.get("aptNm") or "알수없음").strip()
            deposit_val = _parse_price((t.get("deposit") or "0").strip().replace(",", ""))
            monthly_rent = (t.get("monthlyRent") or "0").strip().replace(",", "")
            area = (t.get("excluUseAr") or "0").strip()
            floor = (t.get("floor") or "?").strip()
            dong = (t.get("umdNm") or "").strip()
            deal_day = (t.get("dealDay") or "?").strip()

        deposit_display = f"{deposit_val / 10000:.1f}억원" if isinstance(deposit_val, (int, float)) else f"{deposit_val}만원"

        if monthly_rent == "0" or monthly_rent == "":
            rent_type = "전세"
            price_str = deposit_display
        else:
            rent_type = "월세"
            price_str = f"보증금 {deposit_display} / 월 {monthly_rent}만원"

        return f"- {dong} {apt_name} | {area}㎡ | {floor}층 | [{rent_type}] {price_str} | {year_month[:4]}.{year_month[4:]}.{deal_day}"

    # 보증금순 정렬
    def _get_deposit(t):
        d = t.get("deposit", 0) if from_es else _parse_price((t.get("deposit") or "0").strip().replace(",", ""))
        return d if isinstance(d, (int, float)) else 0

    sorted_rentals = sorted(rentals, key=_get_deposit, reverse=True)
    total = len(sorted_rentals)
    source = "ES 캐시" if from_es else "API"

    # 요약 통계 (전체 기준)
    deposits = [_get_deposit(t) for t in sorted_rentals if _get_deposit(t) > 0]
    stats = ""
    if deposits:
        avg_d = sum(deposits) / len(deposits)
        max_d = max(deposits)
        min_d = min(deposits)
        stats = f"\n■ 보증금 요약 ({total}건 전체 기준): 평균 {avg_d/10000:.1f}억원 | 최고 {max_d/10000:.1f}억원 | 최저 {min_d/10000:.1f}억원\n"

    # 상위 5건
    top_lines = [_format_one_rental(t, from_es) for t in sorted_rentals[:5]]
    # 하위 5건
    bottom_items = sorted_rentals[-5:] if total > 10 else sorted_rentals[5:]
    bottom_lines = [_format_one_rental(t, from_es) for t in reversed(bottom_items)] if bottom_items else []

    actual_total = total_count if total_count > 0 else total
    header = f"📊 {region}({region_code}) {year_month[:4]}년 {year_month[4:]}월 전월세 실거래가 — 전체 {actual_total}건\n{stats}\n"

    body = "▸ 상위 매물\n" + "\n".join(top_lines)
    if bottom_lines:
        body += "\n\n▸ 하위 매물\n" + "\n".join(bottom_lines)

    footer = "\n\n⚠️ 참고: 실거래가 데이터는 신고 지연이 있어 최근 1~2개월은 데이터가 적을 수 있습니다."
    return header + body + footer


# ──────────────────────────────────────────────────────────
# 자동 월 탐색: 데이터 없으면 최대 12개월 이전까지 자동 탐색
# ──────────────────────────────────────────────────────────

MAX_AUTO_SEARCH_MONTHS = 12
# 3개월 이상 전 데이터는 시세 참고용 경고 표시
_STALE_DATA_THRESHOLD_MONTHS = 3


def _months_diff(ym_from: str, ym_to: str) -> int:
    """두 YYYYMM 간 월 차이를 계산합니다."""
    y1, m1 = int(ym_from[:4]), int(ym_from[4:])
    y2, m2 = int(ym_to[:4]), int(ym_to[4:])
    return (y1 - y2) * 12 + (m1 - m2)


def _stale_data_warning(requested_ym: str, actual_ym: str) -> str:
    """요청한 월과 실제 데이터 월의 차이에 따라 경고 메시지를 생성합니다."""
    diff = _months_diff(requested_ym, actual_ym)
    if diff <= 0:
        return ""
    if diff < _STALE_DATA_THRESHOLD_MONTHS:
        return f"⚠️ {requested_ym[:4]}년 {requested_ym[4:]}월에는 데이터가 없어 {actual_ym[:4]}년 {actual_ym[4:]}월 데이터를 조회했습니다.\n\n"
    return (f"⚠️ {requested_ym[:4]}년 {requested_ym[4:]}월 기준 최근 거래가 없어 "
            f"{actual_ym[:4]}년 {actual_ym[4:]}월({diff}개월 전) 데이터를 조회했습니다. "
            f"오래된 데이터이므로 현재 시세와 차이가 있을 수 있습니다. 참고용으로만 활용해주세요.\n\n")


def _prev_year_month(ym: str) -> str:
    """YYYYMM 형식에서 1개월 이전을 반환합니다."""
    dt = datetime(int(ym[:4]), int(ym[4:]), 1) - relativedelta(months=1)
    return dt.strftime("%Y%m")


def _search_trades_with_fallback(region_code: str, year_month: str) -> tuple[list | None, str, bool, int]:
    """매매 데이터를 조회하되, 없으면 최대 12개월 이전까지 자동 탐색합니다.

    Returns:
        (데이터 리스트 또는 None, 실제 조회된 year_month, ES 캐시 여부, 전체 건수)
    """
    ym = year_month
    for i in range(MAX_AUTO_SEARCH_MONTHS):
        # ES 캐시
        cached = _search_trades_from_es(region_code, ym)
        if cached is not None:
            return cached, ym, True, len(cached)
        # API
        result = _fetch_trades_from_api(region_code, ym)
        if isinstance(result, str):
            return None, ym, False, 0  # API 에러
        raw, total_count = result
        if raw:
            _save_trades_to_es(raw, region_code, ym)
            return raw, ym, False, total_count
        # 데이터 없으면 이전 월로
        custom_logger.info(f"매매 데이터 없음: {region_code}/{ym} → 이전 월 탐색 ({i+1}/{MAX_AUTO_SEARCH_MONTHS})")
        ym = _prev_year_month(ym)
    return None, ym, False, 0


def _search_rentals_with_fallback(region_code: str, year_month: str) -> tuple[list | None, str, bool, int]:
    """전월세 데이터를 조회하되, 없으면 최대 12개월 이전까지 자동 탐색합니다."""
    ym = year_month
    for i in range(MAX_AUTO_SEARCH_MONTHS):
        cached = _search_rentals_from_es(region_code, ym)
        if cached is not None:
            return cached, ym, True, len(cached)
        result = _fetch_rentals_from_api(region_code, ym)
        if isinstance(result, str):
            return None, ym, False, 0
        raw, total_count = result
        if raw:
            _save_rentals_to_es(raw, region_code, ym)
            return raw, ym, False, total_count
        custom_logger.info(f"전월세 데이터 없음: {region_code}/{ym} → 이전 월 탐색 ({i+1}/{MAX_AUTO_SEARCH_MONTHS})")
        ym = _prev_year_month(ym)
    return None, ym, False, 0


# ──────────────────────────────────────────────────────────
# @tool 함수 (LangChain 에이전트가 호출)
# ──────────────────────────────────────────────────────────

@tool
def search_apartment_trades(region: str, year_month: str) -> str:
    """아파트 매매 실거래가를 조회합니다.
    해당 월에 데이터가 없으면 최대 12개월 이전까지 자동으로 탐색합니다.

    Args:
        region: 기초자치단체(구/시/군) 이름만 입력. 반드시 "OO구", "OO시", "OO군" 형태여야 합니다. (예: "강남구", "분당구", "해운대구") 광역자치단체(서울, 부산 등)나 동 이름(판교, 잠실 등)은 입력하지 마세요.
        year_month: 조회할 년월 (예: "202501", "202412")

    Returns:
        해당 지역/기간의 아파트 매매 실거래 내역 (최대 10건 요약)
    """
    region_code, error_hint = _resolve_region_code(region)
    if not region_code:
        return error_hint

    data, actual_ym, from_es, total_count = _search_trades_with_fallback(region_code, year_month)

    if data is None:
        # TODO: 향후 한국부동산원 가격지수 API 연동 시, 여기서 보완 데이터 제공
        return f"{region} 최근 {MAX_AUTO_SEARCH_MONTHS}개월간 매매 거래 내역이 없습니다. 거래가 매우 드문 지역일 수 있습니다."

    result = _format_trades(data, region, region_code, actual_ym, from_es=from_es, total_count=total_count)
    warning = _stale_data_warning(year_month, actual_ym)
    if warning:
        result = warning + result
    return result


@tool
def search_apartment_rentals(region: str, year_month: str) -> str:
    """아파트 전월세 실거래가를 조회합니다.
    해당 월에 데이터가 없으면 최대 12개월 이전까지 자동으로 탐색합니다.

    Args:
        region: 기초자치단체(구/시/군) 이름만 입력. 반드시 "OO구", "OO시", "OO군" 형태여야 합니다. (예: "강남구", "분당구", "해운대구") 광역자치단체(서울, 부산 등)나 동 이름(판교, 잠실 등)은 입력하지 마세요.
        year_month: 조회할 년월 (예: "202501", "202412")

    Returns:
        해당 지역/기간의 아파트 전월세 실거래 내역 (최대 10건 요약)
    """
    region_code, error_hint = _resolve_region_code(region)
    if not region_code:
        return error_hint

    data, actual_ym, from_es, total_count = _search_rentals_with_fallback(region_code, year_month)

    if data is None:
        # TODO: 향후 한국부동산원 가격지수 API 연동 시, 여기서 보완 데이터 제공
        return f"{region} 최근 {MAX_AUTO_SEARCH_MONTHS}개월간 전월세 거래 내역이 없습니다. 거래가 매우 드문 지역일 수 있습니다."

    result = _format_rentals(data, region, region_code, actual_ym, from_es=from_es, total_count=total_count)
    warning = _stale_data_warning(year_month, actual_ym)
    if warning:
        result = warning + result
    return result


# ──────────────────────────────────────────────────────────
# 전세가율 계산 내부 함수
# ──────────────────────────────────────────────────────────

def _get_trades_data(region_code: str, year_month: str) -> list:
    """매매 데이터를 자동 탐색으로 가져옵니다 (내부용)."""
    data, actual_ym, from_es, _ = _search_trades_with_fallback(region_code, year_month)
    if data is None:
        return []
    if from_es:
        return data
    # API 원본을 ES 필드명으로 변환
    return [
        {
            "apt_name": (item.get("aptNm") or "").strip(),
            "deal_amount": int(float((item.get("dealAmount") or "0").strip().replace(",", "") or 0)),
            "exclu_use_ar": float((item.get("excluUseAr") or "0").strip() or 0),
            "umd_name": (item.get("umdNm") or "").strip(),
            "cdeal_type": (item.get("cdealType") or "").strip(),
        }
        for item in data
    ]


def _get_rentals_data(region_code: str, year_month: str) -> list:
    """전월세 데이터를 자동 탐색으로 가져옵니다 (내부용)."""
    data, actual_ym, from_es, _ = _search_rentals_with_fallback(region_code, year_month)
    if data is None:
        return []
    if from_es:
        return data
    return [
        {
            "apt_name": (item.get("aptNm") or "").strip(),
            "deposit": int(float((item.get("deposit") or "0").strip().replace(",", "") or 0)),
            "monthly_rent": int(float((item.get("monthlyRent") or "0").strip().replace(",", "") or 0)),
            "exclu_use_ar": float((item.get("excluUseAr") or "0").strip() or 0),
            "umd_name": (item.get("umdNm") or "").strip(),
        }
        for item in data
    ]


@tool
def calculate_jeonse_ratio(region: str, year_month: str) -> str:
    """특정 지역의 전세가율(전세가/매매가 비율)을 계산합니다.
    매매와 전세 데이터를 한 번에 조회하여 전세가율, 매매-전세 갭, 매매/전세 판단 근거를 제공합니다.

    Args:
        region: 기초자치단체(구/시/군) 이름만 입력. 반드시 "OO구", "OO시", "OO군" 형태여야 합니다. (예: "강남구", "분당구", "해운대구") 광역자치단체(서울, 부산 등)나 동 이름(판교, 잠실 등)은 입력하지 마세요.
        year_month: 조회할 년월 (예: "202501", "202412")

    Returns:
        전세가율 분석 결과 (평균 매매가, 평균 전세가, 전세가율, 갭, 판단 근거)
    """
    region_code, error_hint = _resolve_region_code(region)
    if not region_code:
        return error_hint

    # 매매 데이터 조회
    trades = _get_trades_data(region_code, year_month)
    # 해제 거래 필터링
    trades = [t for t in trades if not t.get("cdeal_type")]

    # 전세 데이터 조회 (월세 제외, 전세만)
    rentals = _get_rentals_data(region_code, year_month)
    jeonse_only = [r for r in rentals if r.get("monthly_rent", 0) == 0 and r.get("deposit", 0) > 0]

    if not trades:
        return f"{region} {year_month} 기간에 매매 거래 데이터가 없어 전세가율을 계산할 수 없습니다."
    if not jeonse_only:
        return f"{region} {year_month} 기간에 전세 거래 데이터가 없어 전세가율을 계산할 수 없습니다."

    # 평균 매매가 (만원)
    avg_trade = sum(t["deal_amount"] for t in trades) / len(trades)
    # 평균 전세가 (만원)
    avg_jeonse = sum(r["deposit"] for r in jeonse_only) / len(jeonse_only)

    # 전세가율
    ratio = (avg_jeonse / avg_trade * 100) if avg_trade > 0 else 0
    # 갭 (매매가 - 전세가)
    gap = avg_trade - avg_jeonse

    # 판단 근거
    if ratio >= 70:
        judgment = "전세가율이 70% 이상으로 높은 편입니다. 전세가와 매매가의 차이(갭)가 작아 매매 전환을 고려해볼 만합니다."
    elif ratio >= 50:
        judgment = "전세가율이 50~70% 수준으로 보통입니다. 갭이 적당하여 자금 여력에 따라 매매/전세를 선택할 수 있습니다."
    else:
        judgment = "전세가율이 50% 미만으로 낮은 편입니다. 매매가 대비 전세가가 낮아 전세로 거주하는 것이 자금 효율적일 수 있습니다."

    # 결과 포맷팅
    result = f"""📊 {region}({region_code}) {year_month[:4]}년 {year_month[4:]}월 전세가율 분석

■ 매매 데이터: {len(trades)}건
  평균 매매가: {avg_trade/10000:.1f}억원

■ 전세 데이터: {len(jeonse_only)}건 (전세만, 월세 제외)
  평균 전세가: {avg_jeonse/10000:.1f}억원

■ 전세가율: {ratio:.1f}%
  매매-전세 갭: {gap/10000:.1f}억원

■ 판단 근거
  {judgment}

⚠️ 참고: 전세가율은 평균값 기반이며, 단지·면적·층수별로 차이가 있을 수 있습니다."""

    return result
