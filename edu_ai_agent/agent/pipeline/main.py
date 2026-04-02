"""
부동산 PDF RAG 파이프라인 메인 스크립트.

PDF → 파싱 → 청킹 → 임베딩 → ES 적재

사용법:
  cd edu_ai_agent/agent
  uv run python -m pipeline.main                          # data/ 전체 처리
  uv run python -m pipeline.main pipeline/data/kb_*.pdf   # 특정 파일만
  uv run python -m pipeline.main --recreate-index          # 인덱스 재생성
  uv run python -m pipeline.main --chunk-size 1000         # 청크 크기 조정
"""

import argparse
import sys
from pathlib import Path

from pipeline.config import DATA_DIR
from pipeline.pdf_loader import load_pdf, load_pdfs_from_dir
from pipeline.chunker import chunk_documents
from pipeline.embedder import embed_documents
from pipeline.es_client import create_index, bulk_index


def process_files(files: list[str], chunk_size: int = None, chunk_overlap: int = None):
    """지정된 PDF 파일들을 처리합니다."""
    all_docs = []
    for f in files:
        try:
            docs = load_pdf(f)
            all_docs.extend(docs)
        except Exception as e:
            print(f"  ⚠️ 스킵: {f} — {e}")

    if not all_docs:
        print("처리할 문서가 없습니다.")
        return

    # 청킹
    chunks = chunk_documents(all_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # 임베딩
    embedded = embed_documents(chunks)

    # ES 적재
    bulk_index(embedded)


def main():
    parser = argparse.ArgumentParser(description="부동산 PDF RAG 파이프라인")
    parser.add_argument("files", nargs="*", help="처리할 PDF 파일 경로 (미지정 시 data/ 전체)")
    parser.add_argument("--recreate-index", action="store_true", help="ES 인덱스 재생성")
    parser.add_argument("--chunk-size", type=int, default=None, help="청크 크기 (기본: 500)")
    parser.add_argument("--chunk-overlap", type=int, default=None, help="청크 겹침 (기본: 100)")
    args = parser.parse_args()

    print("=== 부동산 PDF RAG 파이프라인 ===\n")

    # 인덱스 생성/확인
    create_index(recreate=args.recreate_index)

    # PDF 처리
    if args.files:
        print(f"\n📄 지정 파일 처리: {len(args.files)}개")
        process_files(args.files, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
    else:
        print(f"\n📄 전체 처리: {DATA_DIR}")
        all_docs = load_pdfs_from_dir(DATA_DIR)
        if not all_docs:
            return

        chunks = chunk_documents(all_docs, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap)
        embedded = embed_documents(chunks)
        bulk_index(embedded)

    print("\n✅ 파이프라인 완료!")


if __name__ == "__main__":
    main()
