"""DB 설정 모듈 - 스키마/테이블명을 환경변수로 관리"""
import os
import json
from pathlib import Path
from dataclasses import dataclass
from typing import Final, Dict, Any
from functools import lru_cache


@dataclass(frozen=True)
class DbNames:
    """데이터베이스 스키마/테이블명 설정"""
    RAW: str = os.getenv("DB_SCHEMA_RAW", "RawData")
    EMB: str = os.getenv("DB_SCHEMA_EMB", "testcl")
    T_W1: str = os.getenv("TBL_WELCOME_1ST", "welcome_1st")
    T_W2: str = os.getenv("TBL_WELCOME_2ND", "welcome_2nd")
    T_QA: str = os.getenv("TBL_QUICK_ANSWER", "quick_answer")
    T_EMB: str = os.getenv("TBL_PANEL_EMB", "panel_embeddings")
    T_EMB_V: str = "panel_embeddings_v"  # 뷰 이름 (고정)


# 전역 설정 인스턴스
DBN: Final[DbNames] = DbNames()


def fq(schema: str, table: str) -> str:
    """
    Fully Qualified 이름 생성: "Schema"."Table"
    
    Args:
        schema: 스키마 이름
        table: 테이블 이름
        
    Returns:
        완전한 테이블 이름 (예: "RawData"."welcome_1st")
    """
    return f'"{schema}"."{table}"'


# 전처리/가중치/버전 설정
PREPROC_VERSION: Final[str] = os.getenv("PREPROC_VERSION", "v1.0")
KEYWORD_BUNDLE: Final[str] = os.getenv("KEYWORD_BUNDLE", "kr_default_v1")

WEIGHTS: Final[dict[str, float]] = {
    "num": float(os.getenv("WEIGHTS_NUM", "1.0")),
    "cat": float(os.getenv("WEIGHTS_CAT", "0.8")),
    "kw": float(os.getenv("WEIGHTS_KW", "0.8")),
    "len": float(os.getenv("WEIGHTS_LEN", "0.5")),
}

# 키워드 PCA 차원 (None이면 미적용)
KW_PCA_COMPONENTS: Final[str | None] = os.getenv("KW_PCA_COMPONENTS") or None
if KW_PCA_COMPONENTS:
    try:
        _ = int(KW_PCA_COMPONENTS)  # 유효성 검사
    except ValueError:
        KW_PCA_COMPONENTS = None

# 벡터 검색 활성화 여부 (기본값: True, 환경변수로 비활성화 가능)
VECTOR_SEARCH_ENABLED: Final[bool] = os.getenv("VECTOR_SEARCH_ENABLED", "true").lower() in ("true", "1", "yes", "on")

# Pinecone 검색 설정
PINECONE_SEARCH_ENABLED: Final[bool] = os.getenv("PINECONE_SEARCH_ENABLED", "true").lower() in ("true", "1", "yes", "on")
PINECONE_API_KEY: Final[str] = os.getenv("PINECONE_API_KEY", "")
PINECONE_INDEX_NAME: Final[str] = os.getenv("PINECONE_INDEX_NAME", "panel-profiles")
PINECONE_ENVIRONMENT: Final[str] = os.getenv("PINECONE_ENVIRONMENT", "us-east-1")

# 카테고리 설정 - 프로젝트 루트 기준 상대 경로
def _get_project_root() -> Path:
    """프로젝트 루트 디렉토리를 자동으로 찾기"""
    # 현재 파일 위치: server/app/core/config.py
    current_file = Path(__file__).resolve()
    # 프로젝트 루트: server/app/core -> server/app -> server -> 프로젝트 루트
    project_root = current_file.parent.parent.parent.parent
    return project_root

_PROJECT_ROOT = _get_project_root()

# 환경변수가 있으면 사용, 없으면 자동으로 찾기
_category_config_env = os.getenv("CATEGORY_CONFIG_PATH")
if _category_config_env:
    # 환경변수로 절대 경로 또는 상대 경로 지정 가능
    CATEGORY_CONFIG_PATH: Final[str] = _category_config_env
else:
    # 우선순위: 1) notebooks/category_config_수정.json, 2) 프로젝트 루트/category_config_수정.json, 3) notebooks/category_config.json, 4) 프로젝트 루트/category_config.json
    _notebooks_modified_path = _PROJECT_ROOT / "notebooks" / "category_config_수정.json"
    _root_modified_path = _PROJECT_ROOT / "category_config_수정.json"
    _notebooks_path = _PROJECT_ROOT / "notebooks" / "category_config.json"
    _root_path = _PROJECT_ROOT / "category_config.json"
    
    if _notebooks_modified_path.exists():
        CATEGORY_CONFIG_PATH: Final[str] = str(_notebooks_modified_path)
    elif _root_modified_path.exists():
        CATEGORY_CONFIG_PATH: Final[str] = str(_root_modified_path)
    elif _notebooks_path.exists():
        CATEGORY_CONFIG_PATH: Final[str] = str(_notebooks_path)
    elif _root_path.exists():
        CATEGORY_CONFIG_PATH: Final[str] = str(_root_path)
    else:
        # 모두 없으면 notebooks/category_config_수정.json을 기본값으로 (에러는 load_category_config에서 처리)
        CATEGORY_CONFIG_PATH: Final[str] = str(_notebooks_modified_path)

# Pinecone 검색 폴백 설정 (Pinecone 검색 실패 시 기존 검색으로 폴백)
FALLBACK_TO_VECTOR_SEARCH: Final[bool] = os.getenv("FALLBACK_TO_VECTOR_SEARCH", "true").lower() in ("true", "1", "yes", "on")


@lru_cache(maxsize=1)
def load_category_config() -> Dict[str, Any]:
    """
    category_config.json 파일을 로드 (캐싱됨)
    
    Returns:
        카테고리 설정 딕셔너리
        
    Raises:
        FileNotFoundError: 설정 파일이 없을 경우
        json.JSONDecodeError: JSON 파싱 실패 시
    """
    if not os.path.exists(CATEGORY_CONFIG_PATH):
        raise FileNotFoundError(f"카테고리 설정 파일을 찾을 수 없습니다: {CATEGORY_CONFIG_PATH}")
    
    with open(CATEGORY_CONFIG_PATH, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    return config

