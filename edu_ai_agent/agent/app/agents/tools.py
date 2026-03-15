"""
부동산 실거래가 조회 Tool 정의.
- ES 캐시 우선 조회 → 없으면 data.go.kr API 호출 → ES에 저장
- 아파트 매매 실거래가 상세 API (데이터셋 15126468)
- 아파트 전월세 실거래가 API (데이터셋 15126474)

[LangChain 매핑]
- https://docs.langchain.com/oss/python/langchain/quickstart → Step 3: Tool Creation
- https://docs.langchain.com/oss/python/langchain/agents → Tools Management > Static Tools
"""

from datetime import datetime, timezone

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

def _resolve_region_code(region: str) -> str | None:
    """ES에서 지역코드를 검색합니다. ES 미연결 시 폴백 딕셔너리 사용."""
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
                return hits[0]["_source"]["region_code"]
        except Exception as e:
            custom_logger.error(f"ES 지역코드 검색 실패: {e}")

    # 폴백: 하드코딩 딕셔너리
    return _resolve_region_code_fallback(region)


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

def _search_trades_from_es(region_code: str, year_month: str) -> list | None:
    """ES에서 매매 거래 데이터를 조회합니다."""
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
                "size": 30,
                "sort": [{"deal_amount": {"order": "desc"}}],
            },
        )
        hits = result["hits"]["hits"]
        if not hits:
            return None
        return [h["_source"] for h in hits]
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
    """ES에서 전월세 거래 데이터를 조회합니다."""
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
                "size": 30,
                "sort": [{"deposit": {"order": "desc"}}],
            },
        )
        hits = result["hits"]["hits"]
        if not hits:
            return None
        return [h["_source"] for h in hits]
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
        "numOfRows": "30",
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
    items = body.get("items", {})
    if not items or items == "":
        return []

    item_list = items.get("item", [])
    if isinstance(item_list, dict):
        item_list = [item_list]
    return item_list


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
        "numOfRows": "30",
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
    items = body.get("items", {})
    if not items or items == "":
        return []

    item_list = items.get("item", [])
    if isinstance(item_list, dict):
        item_list = [item_list]
    return item_list


# ──────────────────────────────────────────────────────────
# 포맷팅
# ──────────────────────────────────────────────────────────

def _format_trades(trades: list, region: str, region_code: str, year_month: str, from_es: bool = False) -> str:
    """매매 거래 데이터를 사용자에게 보여줄 문자열로 포맷합니다."""
    # 해제 거래 필터링 (ES 데이터는 이미 cdeal_type 필드 사용)
    cdeal_key = "cdeal_type" if from_es else "cdealType"
    valid = [t for t in trades if not t.get(cdeal_key)]

    if not valid:
        return f"{region} {year_month} 기간에 유효한 거래 내역이 없습니다 (모두 해제됨)."

    summaries = []
    for t in valid[:10]:
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

        if isinstance(price_val, (int, float)):
            price_display = f"{price_val / 10000:.1f}억원"
        else:
            price_display = f"{price_val}만원"

        line = (
            f"- {dong} {apt_name} | {area}㎡ | {floor}층 | "
            f"{price_display} | {year_month[:4]}.{year_month[4:]}.{deal_day}"
        )
        if trade_type:
            line += f" | {trade_type}"
        summaries.append(line)

    total = len(valid)
    source = "ES 캐시" if from_es else "API"
    header = f"📊 {region}({region_code}) {year_month[:4]}년 {year_month[4:]}월 매매 실거래가 — 총 {total}건 (상위 10건, {source})\n\n"
    footer = "\n\n⚠️ 참고: 실거래가 데이터는 신고 지연이 있어 최근 1~2개월은 데이터가 적을 수 있습니다."
    return header + "\n".join(summaries) + footer


def _format_rentals(rentals: list, region: str, region_code: str, year_month: str, from_es: bool = False) -> str:
    """전월세 거래 데이터를 사용자에게 보여줄 문자열로 포맷합니다."""
    if not rentals:
        return f"{region} {year_month} 기간에 전월세 거래 내역이 없습니다."

    summaries = []
    for t in rentals[:10]:
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

        if isinstance(deposit_val, (int, float)):
            deposit_display = f"{deposit_val / 10000:.1f}억원"
        else:
            deposit_display = f"{deposit_val}만원"

        if monthly_rent == "0" or monthly_rent == "":
            rent_type = "전세"
            price_str = deposit_display
        else:
            rent_type = "월세"
            price_str = f"보증금 {deposit_display} / 월 {monthly_rent}만원"

        summaries.append(
            f"- {dong} {apt_name} | {area}㎡ | {floor}층 | "
            f"[{rent_type}] {price_str} | {year_month[:4]}.{year_month[4:]}.{deal_day}"
        )

    total = len(rentals)
    source = "ES 캐시" if from_es else "API"
    header = f"📊 {region}({region_code}) {year_month[:4]}년 {year_month[4:]}월 전월세 실거래가 — 총 {total}건 (상위 10건, {source})\n\n"
    footer = "\n\n⚠️ 참고: 실거래가 데이터는 신고 지연이 있어 최근 1~2개월은 데이터가 적을 수 있습니다."
    return header + "\n".join(summaries) + footer


# ──────────────────────────────────────────────────────────
# @tool 함수 (LangChain 에이전트가 호출)
# ──────────────────────────────────────────────────────────

@tool
def search_apartment_trades(region: str, year_month: str) -> str:
    """아파트 매매 실거래가를 조회합니다.

    Args:
        region: 시군구 이름 (예: "강남구", "분당구", "송파구")
        year_month: 조회할 년월 (예: "202501", "202412")

    Returns:
        해당 지역/기간의 아파트 매매 실거래 내역 (최대 10건 요약)
    """
    region_code = _resolve_region_code(region)
    if not region_code:
        return f"'{region}'에 해당하는 지역코드를 찾을 수 없습니다. 시군구 단위로 입력해주세요. (예: 강남구, 분당구)"

    # 1) ES 캐시 조회
    cached = _search_trades_from_es(region_code, year_month)
    if cached is not None:
        custom_logger.info(f"ES 캐시 히트: trades {region_code}/{year_month}")
        return _format_trades(cached, region, region_code, year_month, from_es=True)

    # 2) API 호출
    raw_items = _fetch_trades_from_api(region_code, year_month)
    if isinstance(raw_items, str):
        return raw_items  # 에러 메시지
    if not raw_items:
        return f"{region} {year_month} 기간에 거래 내역이 없습니다."

    # 3) ES에 저장
    _save_trades_to_es(raw_items, region_code, year_month)

    # 4) 포맷팅 반환
    return _format_trades(raw_items, region, region_code, year_month, from_es=False)


@tool
def search_apartment_rentals(region: str, year_month: str) -> str:
    """아파트 전월세 실거래가를 조회합니다.

    Args:
        region: 시군구 이름 (예: "강남구", "분당구", "송파구")
        year_month: 조회할 년월 (예: "202501", "202412")

    Returns:
        해당 지역/기간의 아파트 전월세 실거래 내역 (최대 10건 요약)
    """
    region_code = _resolve_region_code(region)
    if not region_code:
        return f"'{region}'에 해당하는 지역코드를 찾을 수 없습니다. 시군구 단위로 입력해주세요. (예: 강남구, 분당구)"

    # 1) ES 캐시 조회
    cached = _search_rentals_from_es(region_code, year_month)
    if cached is not None:
        custom_logger.info(f"ES 캐시 히트: rentals {region_code}/{year_month}")
        return _format_rentals(cached, region, region_code, year_month, from_es=True)

    # 2) API 호출
    raw_items = _fetch_rentals_from_api(region_code, year_month)
    if isinstance(raw_items, str):
        return raw_items
    if not raw_items:
        return f"{region} {year_month} 기간에 전월세 거래 내역이 없습니다."

    # 3) ES에 저장
    _save_rentals_to_es(raw_items, region_code, year_month)

    # 4) 포맷팅 반환
    return _format_rentals(raw_items, region, region_code, year_month, from_es=False)
