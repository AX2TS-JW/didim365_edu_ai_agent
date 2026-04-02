"""
텍스트 청킹: Document를 검색에 적합한 크기로 분할합니다.
RecursiveCharacterTextSplitter를 사용하여 문맥이 유지되도록 분할.
"""

from langchain_text_splitters import RecursiveCharacterTextSplitter

from pipeline.config import CHUNK_SIZE, CHUNK_OVERLAP


def chunk_documents(documents: list, chunk_size: int = None, chunk_overlap: int = None) -> list:
    """Document 리스트를 청크로 분할합니다.

    Args:
        documents: PDF에서 로드한 Document 리스트
        chunk_size: 청크 크기 (기본: config.CHUNK_SIZE)
        chunk_overlap: 청크 겹침 (기본: config.CHUNK_OVERLAP)

    Returns:
        list[Document]: 분할된 청크 Document 리스트
    """
    chunk_size = chunk_size or CHUNK_SIZE
    chunk_overlap = chunk_overlap or CHUNK_OVERLAP

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)

    # 빈 청크 필터링
    chunks = [c for c in chunks if c.page_content.strip()]

    print(f"  청킹: {len(documents)}페이지 → {len(chunks)}청크 (size={chunk_size}, overlap={chunk_overlap})")
    return chunks
