"""비동기 데이터베이스 세션 관리"""
import os
import logging
from typing import AsyncGenerator
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

logger = logging.getLogger(__name__)


def _build_async_uri() -> str:
    """
    비동기 DB URI 구성 (psycopg 사용)
    
    우선순위:
    1. ASYNC_DATABASE_URI (전체 URI)
    2. PG* / DB_* 환경변수 조합
    
    Note: asyncpg는 Python 3.13 호환성 문제로 psycopg 사용
    """
    # 1) 전체 URI가 들어온 경우 그대로 사용
    uri = os.getenv("ASYNC_DATABASE_URI")
    if uri:
        # asyncpg가 있으면 psycopg로 변환
        if "postgresql+asyncpg" in uri:
            uri = uri.replace("postgresql+asyncpg", "postgresql+psycopg")
        return uri
    
    # 2) PG* / DB_*로 조합 (기본값 포함)
    user = os.getenv("PGUSER") or os.getenv("DB_USER", "postgres")
    pwd = os.getenv("PGPASSWORD") or os.getenv("DB_PASSWORD", "")
    host = os.getenv("PGHOST") or os.getenv("DB_HOST", "localhost")
    port = os.getenv("PGPORT") or os.getenv("DB_PORT", "5432")
    db = os.getenv("PGDATABASE") or os.getenv("DB_NAME", "postgres")
    ssl = os.getenv("PGSSLMODE") or os.getenv("DB_SSLMODE", "require")
    
    # psycopg 스킴 사용 (asyncpg는 Python 3.13 호환성 문제)
    return f"postgresql+psycopg://{user}:{pwd}@{host}:{port}/{db}?sslmode={ssl}"


USE_NULL_POOL = os.getenv("DB_USE_NULL_POOL", "false").lower() == "true"

ASYNC_URI = _build_async_uri()

# 유효하지 않은 URI 체크
if not ASYNC_URI or ":///" in ASYNC_URI or "://@/" in ASYNC_URI:
    engine = None
    SessionLocal = None
    print("[WARNING] ASYNC_DATABASE_URI not configured. Database features will be disabled.")
else:
    # 비동기 엔진 생성
    engine = create_async_engine(
        ASYNC_URI,
        echo=False,
        pool_pre_ping=True,
        poolclass=NullPool if USE_NULL_POOL else None,  # Neon이면 true 권장
    )
    
    # 비동기 세션 팩토리 생성
    SessionLocal = async_sessionmaker(
        bind=engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    pool_type = "NullPool" if USE_NULL_POOL else "QueuePool"


async def get_session() -> AsyncGenerator[AsyncSession, None]:
    """
    비동기 데이터베이스 세션 의존성
    
    RawData 스키마를 우선 탐색 경로로 고정
    
    Yields:
        AsyncSession: 비동기 SQLAlchemy 세션
        
    Raises:
        RuntimeError: 세션 생성 실패 시
    """
    if SessionLocal is None:
        raise RuntimeError("Database not configured. Please set ASYNC_DATABASE_URI in .env")
    
    session = SessionLocal()
    
    try:
        # RawData 우선 탐색 경로 고정
        await session.execute(text('SET search_path TO "RawData", public'))
        yield session  # FastAPI가 자동으로 cleanup 처리
    except Exception as e:
        await session.rollback()
        logger.error(f"세션 오류: {str(e)}", exc_info=True)
        raise
    finally:
        await session.close()
