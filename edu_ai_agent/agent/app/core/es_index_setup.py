"""Elasticsearch 인덱스 매핑 정의 및 자동 생성."""

from app.core.es_client import es_client
from app.utils.logger import custom_logger

# 인덱스 이름
IDX_REGION_CODES = "edu_region_codes"
IDX_APT_TRADES = "edu_apt_trades"
IDX_APT_RENTALS = "edu_apt_rentals"

# 법정동코드
REGION_CODES_MAPPING = {
    "mappings": {
        "properties": {
            "region_name": {"type": "keyword"},
            "region_code": {"type": "keyword"},
            "city": {"type": "keyword"},
            "aliases": {"type": "text", "analyzer": "standard"},
        }
    }
}

# 매매 실거래가
APT_TRADES_MAPPING = {
    "mappings": {
        "properties": {
            "region_code": {"type": "keyword"},
            "year_month": {"type": "keyword"},
            "apt_name": {"type": "text"},
            "deal_amount": {"type": "long"},
            "exclu_use_ar": {"type": "float"},
            "floor": {"type": "keyword"},
            "umd_name": {"type": "keyword"},
            "deal_day": {"type": "keyword"},
            "dealing_gbn": {"type": "keyword"},
            "cdeal_type": {"type": "keyword"},
            "fetched_at": {"type": "date"},
        }
    }
}

# 전월세 실거래가
APT_RENTALS_MAPPING = {
    "mappings": {
        "properties": {
            "region_code": {"type": "keyword"},
            "year_month": {"type": "keyword"},
            "apt_name": {"type": "text"},
            "deposit": {"type": "long"},
            "monthly_rent": {"type": "long"},
            "exclu_use_ar": {"type": "float"},
            "floor": {"type": "keyword"},
            "umd_name": {"type": "keyword"},
            "deal_day": {"type": "keyword"},
            "fetched_at": {"type": "date"},
        }
    }
}

INDEX_CONFIGS = {
    IDX_REGION_CODES: REGION_CODES_MAPPING,
    IDX_APT_TRADES: APT_TRADES_MAPPING,
    IDX_APT_RENTALS: APT_RENTALS_MAPPING,
}


def ensure_indices():
    """인덱스가 없으면 생성합니다."""
    if es_client is None:
        custom_logger.warning("ES 미연결 — 인덱스 생성 건너뜀")
        return

    for index_name, mapping in INDEX_CONFIGS.items():
        try:
            if not es_client.indices.exists(index=index_name):
                es_client.indices.create(index=index_name, body=mapping)
                custom_logger.info(f"인덱스 생성 완료: {index_name}")
            else:
                custom_logger.info(f"인덱스 이미 존재: {index_name}")
        except Exception as e:
            custom_logger.error(f"인덱스 생성 실패 ({index_name}): {e}")
