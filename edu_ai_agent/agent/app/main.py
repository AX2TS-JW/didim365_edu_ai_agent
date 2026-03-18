import time
from contextlib import asynccontextmanager
from fastapi import FastAPI, APIRouter, Request
from fastapi.middleware.cors import CORSMiddleware
from app.core.config import settings
from app.api.routes.threads import threads_router
from app.api.routes.chat import chat_router
from app.utils.logger import custom_logger


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup: Opik 초기화
    opik_settings = settings.OPIK
    if opik_settings and opik_settings.URL_OVERRIDE:
        import os
        os.environ["OPIK_URL_OVERRIDE"] = opik_settings.URL_OVERRIDE
        os.environ["OPIK_WORKSPACE"] = opik_settings.WORKSPACE or "default"
        os.environ["OPIK_PROJECT_NAME"] = opik_settings.PROJECT or "default"
        if opik_settings.API_KEY:
            os.environ["OPIK_API_KEY"] = opik_settings.API_KEY
        custom_logger.info(f"Opik 연결 완료: {opik_settings.URL_OVERRIDE}")

    # startup: ES 인덱스 자동 생성
    from app.core.es_index_setup import ensure_indices
    ensure_indices()
    yield
    # shutdown


app = FastAPI(
    title="Edu Agent Template",
    description="LangChain 기반 에이전트 교육용 템플릿",
    version="0.1.0",
    lifespan=lifespan,
)

api_router = APIRouter(prefix=settings.API_V1_PREFIX)


# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 라우터 등록
api_router.include_router(threads_router, tags=["threads"])
api_router.include_router(chat_router, tags=["chat"])

app.include_router(api_router)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    custom_logger.info(f"➡️ 요청 시작: {request.method} {request.url.path}")
    start_time = time.time()

    response = await call_next(request)

    process_time = time.time() - start_time
    custom_logger.info(
        f"⬅️ 요청 종료: {request.method} {request.url.path} "
        f"(실행 시간: {process_time:.3f}초) "
        f"상태코드: {response.status_code}"
    )

    return response


@app.get("/")
async def root():
    return {"message": "Edu Agent API", "version": "0.1.0"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )