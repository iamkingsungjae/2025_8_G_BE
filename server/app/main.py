"""
Panel Insight API - Main Application Entry Point
슬림화된 버전 (라우터 등록 및 핵심 설정만)
"""
import sys
import asyncio

# Windows에서 ProactorEventLoop 문제 해결 (모든 import 전에 설정)
if sys.platform == 'win32':
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception as e:
        pass  # 실패해도 계속 진행

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
from pathlib import Path
from dotenv import load_dotenv
import os
import logging
import traceback

# 로깅 설정 (프로덕션 환경: INFO 레벨, 개발 환경: 환경 변수로 제어)
log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# HTTP 관련 로그는 WARNING으로 유지
logging.getLogger('urllib3.connectionpool').setLevel(logging.WARNING)
logging.getLogger('httpcore').setLevel(logging.WARNING)
logging.getLogger('httpx').setLevel(logging.WARNING)

# 환경 변수 로드 - Path 사용으로 경로 안정화
ENV_PATH = (Path(__file__).resolve().parents[1] / ".env")
load_dotenv(ENV_PATH, override=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    앱 수명주기 관리 (startup/shutdown)
    """
    # startup
    yield
    
    # shutdown


# FastAPI 앱 초기화 (lifespan 포함)
app = FastAPI(title="Panel Insight API", version="0.1.0", lifespan=lifespan)

# 요청 미들웨어: 모든 요청 로깅
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """모든 요청을 로깅하는 미들웨어"""
    import time
    start_time = time.time()
    logger = logging.getLogger(__name__)
    
    try:
        response = await call_next(request)
        process_time = time.time() - start_time
        # 에러 응답만 로깅
        if response.status_code >= 400:
            logger.warning(f"[REQUEST] {request.method} {request.url.path} - 응답: {response.status_code} ({process_time:.2f}초)")
        return response
    except Exception as e:
        process_time = time.time() - start_time
        logger.error(f"[REQUEST] {request.method} {request.url.path} - 에러 발생 ({process_time:.2f}초): {e}", exc_info=True)
        raise

# 전역 예외 핸들러 (개발 환경에서 상세 오류 정보 반환)
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """전역 예외 핸들러 - 모든 예외를 잡아서 상세 정보 반환"""
    error_detail = traceback.format_exc()
    logger = logging.getLogger(__name__)
    logger.error(f"전역 예외 발생: {exc}\n{error_detail}", exc_info=True)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": str(exc),
            "detail": error_detail,
            "path": str(request.url),
            "method": request.method
        }
    )

# CORS 설정 (개발 환경 - 모든 origin 허용)
# OPTIONS 요청 처리를 위해 더 관대한 설정 사용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 개발 환경에서는 모든 origin 허용
    allow_credentials=True,  # credentials 허용으로 변경
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "HEAD", "PATCH"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,  # preflight 요청 캐시 시간
)

# API 라우터 등록
from app.api.search import router as search_router
from app.api.panels import router as panels_router
from app.api.health import router as health_router
from app.api.clustering import router as clustering_router
from app.api.clustering_viz import router as clustering_viz_router
from app.api.precomputed import router as precomputed_router
app.include_router(search_router)
app.include_router(panels_router)
app.include_router(health_router)
app.include_router(clustering_router)
app.include_router(clustering_viz_router)
app.include_router(precomputed_router)

# 기본 라우트
@app.get("/")
def root():
    return {"name": "Panel Insight API", "version": "0.1.0"}
