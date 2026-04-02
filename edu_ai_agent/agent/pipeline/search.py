"""
검색 테스트: BM25 / Vector / Hybrid 검색을 테스트합니다.

사용법:
  cd edu_ai_agent/agent
  uv run python -m pipeline.search "강남구 시장 전망"
  uv run python -m pipeline.search "금리 인상 영향" --mode bm25
  uv run python -m pipeline.search "전세 시장 동향" --mode hybrid --top-k 3
"""

import argparse

from pipeline.config import ES_INDEX
from pipeline.es_client import get_es_client
from pipeline.embedder import embed_query


def search_bm25(query: str, top_k: int = 5) -> list[dict]:
    """BM25 키워드 검색."""
    es = get_es_client()
    result = es.search(
        index=ES_INDEX,
        body={
            "query": {"match": {"content": query}},
            "size": top_k,
        },
    )
    return _format_hits(result["hits"]["hits"])


def search_vector(query: str, top_k: int = 5) -> list[dict]:
    """kNN 벡터 검색."""
    es = get_es_client()
    query_vector = embed_query(query)
    result = es.search(
        index=ES_INDEX,
        body={
            "knn": {
                "field": "content_vector",
                "query_vector": query_vector,
                "k": top_k,
                "num_candidates": top_k * 10,
            },
            "size": top_k,
        },
    )
    return _format_hits(result["hits"]["hits"])


def search_hybrid(query: str, top_k: int = 5) -> list[dict]:
    """BM25 + kNN 하이브리드 검색 (score 합산 방식, RRF 라이선스 불필요)."""
    es = get_es_client()
    query_vector = embed_query(query)
    result = es.search(
        index=ES_INDEX,
        body={
            "query": {"match": {"content": query}},
            "knn": {
                "field": "content_vector",
                "query_vector": query_vector,
                "k": top_k,
                "num_candidates": top_k * 10,
            },
            "size": top_k,
        },
    )
    return _format_hits(result["hits"]["hits"])


def _format_hits(hits: list) -> list[dict]:
    """ES 검색 결과를 정리합니다."""
    results = []
    for hit in hits:
        source = hit["_source"]
        results.append({
            "score": hit["_score"],
            "content": source.get("content", "")[:300],
            "source_file": source.get("metadata", {}).get("source_file", "unknown"),
            "page": source.get("metadata", {}).get("page", "?"),
        })
    return results


def main():
    parser = argparse.ArgumentParser(description="부동산 PDF 검색 테스트")
    parser.add_argument("query", help="검색 쿼리")
    parser.add_argument("--mode", choices=["bm25", "vector", "hybrid"], default="hybrid")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    search_fn = {"bm25": search_bm25, "vector": search_vector, "hybrid": search_hybrid}[args.mode]

    print(f"=== {args.mode.upper()} 검색: '{args.query}' (top-{args.top_k}) ===\n")
    results = search_fn(args.query, args.top_k)

    for i, r in enumerate(results, 1):
        print(f"[{i}] score={r['score']:.4f} | {r['source_file']} (p.{r['page']})")
        print(f"    {r['content'][:150]}...")
        print()

    if not results:
        print("검색 결과가 없습니다.")


if __name__ == "__main__":
    main()
