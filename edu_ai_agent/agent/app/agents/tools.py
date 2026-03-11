"""
부동산 실거래가 조회 Tool 정의.
- 아파트 매매 실거래가 상세 API (데이터셋 15126468)
- 아파트 전월세 실거래가 API (데이터셋 15126474)

[LangChain 매핑]
- https://docs.langchain.com/oss/python/langchain/quickstart → Step 3: Tool Creation
- https://docs.langchain.com/oss/python/langchain/agents → Tools Management > Static Tools
"""

import httpx
import xmltodict
from langchain.tools import tool

from app.core.config import settings
from app.utils.logger import custom_logger

# 주요 지역 → 법정동코드(5자리) 매핑
# 실거래가 API는 시군구코드(5자리)를 필수 파라미터로 요구합니다.
REGION_CODE_MAP = {
    # 서울
    "강남구": "11680", "서초구": "11650", "송파구": "11710",
    "강동구": "11740", "마포구": "11440", "용산구": "11170",
    "성동구": "11200", "광진구": "11215", "동작구": "11590",
    "영등포구": "11560", "강서구": "11500", "양천구": "11470",
    "구로구": "11530", "금천구": "11545", "관악구": "11620",
    "노원구": "11350", "도봉구": "11320", "강북구": "11305",
    "성북구": "11290", "종로구": "11110", "중구": "11140",
    "중랑구": "11260", "동대문구": "11230", "은평구": "11380",
    "서대문구": "11410",
    # 경기
    "분당구": "41135", "수정구": "41131", "중원구": "41133",
    "수지구": "41465", "기흥구": "41463", "처인구": "41461",
    "수원시": "41110", "고양시": "41280", "용인시": "41460",
    "성남시": "41130", "화성시": "41590", "평택시": "41220",
}


def _resolve_region_code(region: str) -> str | None:
    """지역명에서 법정동 코드를 찾습니다. 부분 매칭을 지원합니다."""
    # 정확히 일치
    if region in REGION_CODE_MAP:
        return REGION_CODE_MAP[region]
    # 부분 매칭 (e.g. "강남" → "강남구")
    for name, code in REGION_CODE_MAP.items():
        if region in name or name in region:
            return code
    return None


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

    api_key = settings.DATA_GO_KR_API_KEY
    if not api_key:
        return "data.go.kr API 키가 설정되지 않았습니다. .env에 DATA_GO_KR_API_KEY를 추가해주세요."

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

    # 응답 구조 파싱
    body = data.get("response", {}).get("body", {})
    items = body.get("items", {})

    if not items or items == "":
        return f"{region} {year_month} 기간에 거래 내역이 없습니다."

    item_list = items.get("item", [])
    if isinstance(item_list, dict):
        item_list = [item_list]  # 단건일 경우 리스트로 변환

    # 해제(취소) 거래 필터링 — cdealType이 None이 아니면 해제된 거래
    valid_trades = [
        t for t in item_list
        if not t.get("cdealType")
    ]

    if not valid_trades:
        return f"{region} {year_month} 기간에 유효한 거래 내역이 없습니다 (모두 해제됨)."

    # 상위 10건 요약
    # API 필드: aptNm(아파트명), dealAmount(거래금액), excluUseAr(전용면적),
    #           floor(층), umdNm(법정동), dealDay(일), dealingGbn(거래유형)
    summaries = []
    for t in valid_trades[:10]:
        apt_name = (t.get("aptNm") or "알수없음").strip()
        price = (t.get("dealAmount") or "0").strip().replace(",", "")
        area = (t.get("excluUseAr") or "0").strip()
        floor = (t.get("floor") or "?").strip()
        dong = (t.get("umdNm") or "").strip()
        deal_day = (t.get("dealDay") or "?").strip()
        trade_type = (t.get("dealingGbn") or "").strip()

        price_eok = int(price) / 10000 if price.isdigit() else price
        summaries.append(
            f"- {dong} {apt_name} | {area}㎡ | {floor}층 | "
            f"{price_eok:.1f}억원 | {year_month[:4]}.{year_month[4:]}.{deal_day}"
            + (f" | {trade_type}" if trade_type else "")
        )

    total = len(valid_trades)
    header = f"📊 {region}({region_code}) {year_month[:4]}년 {year_month[4:]}월 매매 실거래가 — 총 {total}건 (상위 10건 표시)\n\n"
    return header + "\n".join(summaries)


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

    api_key = settings.DATA_GO_KR_API_KEY
    if not api_key:
        return "data.go.kr API 키가 설정되지 않았습니다. .env에 DATA_GO_KR_API_KEY를 추가해주세요."

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

    # 응답 구조 파싱
    body = data.get("response", {}).get("body", {})
    items = body.get("items", {})

    if not items or items == "":
        return f"{region} {year_month} 기간에 전월세 거래 내역이 없습니다."

    item_list = items.get("item", [])
    if isinstance(item_list, dict):
        item_list = [item_list]

    # 상위 10건 요약
    # API 필드: aptNm(단지명), deposit(보증금), monthlyRent(월세, 0이면 전세),
    #           excluUseAr(전용면적), floor(층), umdNm(법정동), dealDay(일)
    summaries = []
    for t in item_list[:10]:
        apt_name = (t.get("aptNm") or "알수없음").strip()
        deposit = (t.get("deposit") or "0").strip().replace(",", "")
        monthly_rent = (t.get("monthlyRent") or "0").strip().replace(",", "")
        area = (t.get("excluUseAr") or "0").strip()
        floor = (t.get("floor") or "?").strip()
        dong = (t.get("umdNm") or "").strip()
        deal_day = (t.get("dealDay") or "?").strip()

        # 전세/월세 구분
        deposit_eok = int(deposit) / 10000 if deposit.isdigit() else deposit
        if monthly_rent == "0" or monthly_rent == "":
            rent_type = "전세"
            price_str = f"{deposit_eok:.1f}억원"
        else:
            rent_type = "월세"
            price_str = f"보증금 {deposit_eok:.1f}억원 / 월 {monthly_rent}만원"

        summaries.append(
            f"- {dong} {apt_name} | {area}㎡ | {floor}층 | "
            f"[{rent_type}] {price_str} | {year_month[:4]}.{year_month[4:]}.{deal_day}"
        )

    total = len(item_list)
    header = f"📊 {region}({region_code}) {year_month[:4]}년 {year_month[4:]}월 전월세 실거래가 — 총 {total}건 (상위 10건 표시)\n\n"
    return header + "\n".join(summaries)
