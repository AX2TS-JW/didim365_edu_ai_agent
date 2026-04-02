"""
PDF 파싱: PDF 파일에서 페이지별 텍스트를 추출합니다.
PyPDFLoader를 사용하여 Document 객체 리스트로 반환.
"""

from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader


def load_pdf(file_path: str) -> list:
    """PDF 파일을 페이지별 Document 리스트로 로드합니다.

    Args:
        file_path: PDF 파일 경로

    Returns:
        list[Document]: 페이지별 Document 객체 (page_content + metadata)
    """
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF 파일이 없습니다: {file_path}")
    if not path.suffix.lower() == ".pdf":
        raise ValueError(f"PDF 파일이 아닙니다: {file_path}")

    loader = PyPDFLoader(str(path))
    documents = loader.load()

    # 빈 페이지 필터링
    documents = [doc for doc in documents if doc.page_content.strip()]

    # 소스 파일명을 metadata에 추가
    for doc in documents:
        doc.metadata["source_file"] = path.name
        doc.metadata["source_path"] = str(path)

    print(f"  PDF 로드: {path.name} → {len(documents)}페이지")
    return documents


def load_pdfs_from_dir(data_dir: str) -> list:
    """디렉토리 내 모든 PDF를 로드합니다.

    Args:
        data_dir: PDF 파일들이 있는 디렉토리

    Returns:
        list[Document]: 전체 페이지 Document 리스트
    """
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"디렉토리가 없습니다: {data_dir}")

    pdf_files = sorted(data_path.glob("*.pdf"))
    if not pdf_files:
        print(f"  PDF 파일이 없습니다: {data_dir}")
        return []

    all_docs = []
    for pdf_file in pdf_files:
        try:
            docs = load_pdf(str(pdf_file))
            all_docs.extend(docs)
        except Exception as e:
            print(f"  ⚠️ 로드 실패: {pdf_file.name} — {e}")

    print(f"  총 {len(all_docs)}페이지 로드 ({len(pdf_files)}개 PDF)")
    return all_docs
