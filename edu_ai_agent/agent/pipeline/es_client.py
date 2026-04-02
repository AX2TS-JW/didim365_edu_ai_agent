"""
Elasticsearch 적재: 임베딩된 청크를 ES에 벌크 저장합니다.
인덱스 매핑: content(text) + content_vector(dense_vector) + metadata
"""

from elasticsearch import Elasticsearch

from pipeline.config import ES_URL, ES_USER, ES_PASSWORD, ES_INDEX, EMBEDDING_MODEL

# 임베딩 모델별 차원 수
_EMBEDDING_DIMS = {
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
    "text-embedding-ada-002": 1536,
}

_es_client = None


def get_es_client() -> Elasticsearch:
    """ES 클라이언트 싱글턴을 반환합니다."""
    global _es_client
    if _es_client is None:
        _es_client = Elasticsearch(
            ES_URL,
            basic_auth=(ES_USER, ES_PASSWORD),
            verify_certs=True,
        )
        info = _es_client.info()
        print(f"  ES 연결: {info['version']['number']}")
    return _es_client


def create_index(recreate: bool = False):
    """ES 인덱스를 생성합니다. recreate=True면 기존 인덱스 삭제 후 재생성."""
    es = get_es_client()
    dims = _EMBEDDING_DIMS.get(EMBEDDING_MODEL, 1536)

    if es.indices.exists(index=ES_INDEX):
        if recreate:
            es.indices.delete(index=ES_INDEX)
            print(f"  인덱스 삭제: {ES_INDEX}")
        else:
            print(f"  인덱스 존재: {ES_INDEX} (재생성 안 함)")
            return

    mapping = {
        "mappings": {
            "properties": {
                "content": {"type": "text", "analyzer": "standard"},
                "content_vector": {
                    "type": "dense_vector",
                    "dims": dims,
                    "index": True,
                    "similarity": "cosine",
                },
                "metadata": {
                    "properties": {
                        "source_file": {"type": "keyword"},
                        "page": {"type": "integer"},
                    }
                },
            }
        }
    }

    es.indices.create(index=ES_INDEX, body=mapping)
    print(f"  인덱스 생성: {ES_INDEX} (dims={dims})")


def bulk_index(documents: list[dict]):
    """임베딩된 문서를 ES에 벌크 적재합니다.

    Args:
        documents: [{"content": str, "content_vector": list, "metadata": dict}, ...]
    """
    es = get_es_client()

    if not documents:
        print("  적재할 문서가 없습니다.")
        return

    actions = []
    for i, doc in enumerate(documents):
        actions.append({"index": {"_index": ES_INDEX, "_id": f"{ES_INDEX}_{i}"}})
        actions.append({
            "content": doc["content"],
            "content_vector": doc["content_vector"],
            "metadata": doc.get("metadata", {}),
        })

    # 1000건씩 벌크
    batch_size = 1000
    total = len(documents)
    for start in range(0, len(actions), batch_size * 2):
        batch = actions[start:start + batch_size * 2]
        es.bulk(body=batch, refresh="wait_for")
        indexed = min((start // 2) + batch_size, total)
        print(f"  ES 적재: {indexed}/{total}")

    print(f"  ES 적재 완료: {total}건 → {ES_INDEX}")
