"""
임베딩 생성: 청크 텍스트를 벡터로 변환합니다.
OpenAI text-embedding-3-small 모델 사용.
"""

import time
from langchain_openai import OpenAIEmbeddings

from pipeline.config import OPENAI_API_KEY, EMBEDDING_MODEL


_embeddings = None


def get_embeddings() -> OpenAIEmbeddings:
    """OpenAI 임베딩 모델 싱글턴을 반환합니다."""
    global _embeddings
    if _embeddings is None:
        _embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            openai_api_key=OPENAI_API_KEY,
        )
    return _embeddings


def embed_documents(chunks: list, batch_size: int = 100) -> list[dict]:
    """청크 리스트를 임베딩하여 ES 적재용 dict 리스트로 반환합니다.

    Args:
        chunks: 분할된 Document 리스트
        batch_size: 한 번에 임베딩할 청크 수

    Returns:
        list[dict]: ES 적재용 문서 리스트
            [{"content": str, "content_vector": list[float], "metadata": dict}, ...]
    """
    embeddings_model = get_embeddings()
    results = []
    total = len(chunks)

    for i in range(0, total, batch_size):
        batch = chunks[i:i + batch_size]
        texts = [c.page_content for c in batch]

        # 재시도 로직
        for attempt in range(3):
            try:
                vectors = embeddings_model.embed_documents(texts)
                break
            except Exception as e:
                if attempt < 2:
                    print(f"  ⚠️ 임베딩 재시도 ({attempt + 1}/3): {e}")
                    time.sleep(2 ** attempt)
                else:
                    raise

        for chunk, vector in zip(batch, vectors):
            results.append({
                "content": chunk.page_content,
                "content_vector": vector,
                "metadata": chunk.metadata,
            })

        print(f"  임베딩: {min(i + batch_size, total)}/{total} 완료")

    return results


def embed_query(query: str) -> list[float]:
    """검색 쿼리를 벡터로 변환합니다."""
    return get_embeddings().embed_query(query)
