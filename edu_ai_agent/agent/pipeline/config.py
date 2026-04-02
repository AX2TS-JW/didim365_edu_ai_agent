"""
파이프라인 설정.
환경변수 또는 기본값으로 ES/OpenAI/청크 파라미터를 관리합니다.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Elasticsearch
ES_URL = os.getenv("ES_URL", "https://elasticsearch-edu.didim365.app")
ES_USER = os.getenv("ES_USER", "elastic")
ES_PASSWORD = os.getenv("ES_PASSWORD", "FJl79PA7mMIJajxB1OHgdLEe")
ES_INDEX = os.getenv("ES_INDEX", "edu-real-estate-reports")

# OpenAI Embedding
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# 청크 설정
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# Cohere ReRank (선택)
COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")

# 데이터 디렉토리
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
