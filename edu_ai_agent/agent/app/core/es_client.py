"""Elasticsearch 클라이언트 싱글턴."""

from elasticsearch import Elasticsearch
from app.core.config import settings
from app.utils.logger import custom_logger


def create_es_client() -> Elasticsearch | None:
    """settings 기반으로 ES 클라이언트를 생성합니다."""
    if not settings.ES_URL:
        custom_logger.warning("ES_URL이 설정되지 않아 Elasticsearch를 사용하지 않습니다.")
        return None

    client = Elasticsearch(
        settings.ES_URL,
        basic_auth=(settings.ES_USER, settings.ES_PASSWORD),
        verify_certs=True,
        request_timeout=10,
    )
    try:
        info = client.info()
        custom_logger.info(f"Elasticsearch 연결 성공: {info['version']['number']}")
    except Exception as e:
        custom_logger.error(f"Elasticsearch 연결 실패: {e}")
        return None

    return client


es_client: Elasticsearch | None = create_es_client()
