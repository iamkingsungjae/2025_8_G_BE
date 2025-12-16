"""패널 검색 API 엔드포인트"""
from fastapi import APIRouter, HTTPException
from typing import Dict, Any, List, Optional
from datetime import datetime
import json
import logging
import asyncio
from app.core.config import (
    PINECONE_SEARCH_ENABLED,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
    ANTHROPIC_API_KEY,
    OPENAI_API_KEY,
    load_category_config
)
from app.services.pinecone_filter_converter import PineconeFilterConverter
from app.api.pinecone_panel_details import _get_panel_details_from_pinecone

logger = logging.getLogger(__name__)

router = APIRouter()


@router.get("/api/search/status")
async def get_search_status():
    """검색 상태 확인 (Pinecone 검색 + 필터 검색)"""
    try:
        status = {
            "pinecone_search_enabled": PINECONE_SEARCH_ENABLED,
            "status": "active",
            "message": ""
        }
        
        # Pinecone 검색 상태 확인
        if PINECONE_SEARCH_ENABLED:
            try:
                if PINECONE_API_KEY:
                    # 실제 Pinecone 연결 테스트
                    try:
                        from pinecone import Pinecone
                        pc = Pinecone(api_key=PINECONE_API_KEY)
                        index = pc.Index(PINECONE_INDEX_NAME)
                        
                        # 인덱스 통계 조회로 연결 확인
                        stats = index.describe_index_stats()
                        
                        status["pinecone_available"] = True
                        status["pinecone_index_name"] = PINECONE_INDEX_NAME
                        status["pinecone_connected"] = True
                        status["pinecone_stats"] = {
                            "dimension": stats.get("dimension", "unknown"),
                            "total_vector_count": stats.get("total_vector_count", 0),
                            "index_fullness": stats.get("index_fullness", 0)
                        }
                    except Exception as conn_error:
                        status["pinecone_available"] = False
                        status["pinecone_connected"] = False
                        status["pinecone_error"] = f"Pinecone 연결 실패: {str(conn_error)}"
                        status["pinecone_index_name"] = PINECONE_INDEX_NAME
                else:
                    status["pinecone_available"] = False
                    status["pinecone_connected"] = False
                    status["pinecone_error"] = "Pinecone API 키가 설정되지 않았습니다"
                
                # 카테고리 설정 파일 확인
                try:
                    category_config = load_category_config()
                    status["category_config_loaded"] = True
                    status["category_count"] = len(category_config)
                except Exception as e:
                    status["category_config_loaded"] = False
                    status["category_config_error"] = str(e)
                    
            except Exception as e:
                status["pinecone_available"] = False
                status["pinecone_connected"] = False
                status["pinecone_error"] = str(e)
        else:
            status["pinecone_available"] = False
            status["pinecone_connected"] = False
        
        # 전체 상태 메시지 생성
        modes = []
        if PINECONE_SEARCH_ENABLED and status.get("pinecone_available"):
            modes.append("Pinecone 검색")
        modes.append("필터 검색")
        
        if modes:
            status["message"] = f"사용 가능한 검색 모드: {', '.join(modes)}"
        else:
            status["status"] = "warning"
            status["message"] = "활성화된 검색 모드가 없습니다."
        
        return status
    except Exception as e:
        return {
            "pinecone_search_enabled": False,
            "status": "error",
            "message": f"상태 확인 중 오류: {str(e)}"
        }


@router.get("/api/search/cache")
async def get_cache_status():
    """검색 캐시 상태 확인"""
    import threading
    global _pinecone_cache, _cache_lock
    
    if _cache_lock is None:
        _cache_lock = threading.Lock()
    
    with _cache_lock:
        cache_keys = list(_pinecone_cache.keys())
        cache_size = len(_pinecone_cache)
        cache_max_size = _cache_max_size
        
        # 캐시 항목 요약 정보
        cache_items = []
        for key, value in _pinecone_cache.items():
            cache_items.append({
                "key": key,
                "query": key.split(":")[0] if ":" in key else key,
                "top_k": value.get("top_k", "unknown"),
                "result_count": len(value.get("results", [])),
                "timestamp": value.get("timestamp", "unknown")
            })
        
        return {
            "cache_enabled": True,
            "cache_size": cache_size,
            "cache_max_size": cache_max_size,
            "cache_usage_percent": round((cache_size / cache_max_size) * 100, 2) if cache_max_size > 0 else 0,
            "cache_keys": cache_keys,
            "cache_items": cache_items
        }


@router.delete("/api/search/cache")
async def clear_cache():
    """검색 캐시 초기화"""
    import threading
    global _pinecone_cache, _cache_lock
    
    if _cache_lock is None:
        _cache_lock = threading.Lock()
    
    with _cache_lock:
        cache_size_before = len(_pinecone_cache)
        _pinecone_cache.clear()
        logger.info(f"[Cache] 캐시 초기화 완료: {cache_size_before}개 항목 삭제")
        
        return {
            "success": True,
            "message": f"캐시가 초기화되었습니다. ({cache_size_before}개 항목 삭제)",
            "cache_size_before": cache_size_before,
            "cache_size_after": 0
        }


# 파이프라인 싱글톤 (재사용)
_pipeline_instance: Optional[Any] = None
_pipeline_lock = None

# Pinecone 검색 결과 캐시 (쿼리 -> mb_sn 리스트)
_pinecone_cache: Dict[str, Dict[str, Any]] = {}
_cache_lock = None
_cache_max_size = 100  # 최대 캐시 크기


def _get_cache_key(query: str, top_k: int) -> str:
    """캐시 키 생성"""
    return f"{query.strip().lower()}:{top_k}"


def _get_cached_result(query: str, top_k: int) -> Optional[List[str]]:
    """캐시에서 검색 결과 가져오기"""
    import threading
    global _pinecone_cache, _cache_lock
    
    if _cache_lock is None:
        _cache_lock = threading.Lock()
    
    cache_key = _get_cache_key(query, top_k)
    
    with _cache_lock:
        if cache_key in _pinecone_cache:
            cached = _pinecone_cache[cache_key]
            return cached.get('results')
    
    return None


def _set_cached_result(query: str, top_k: int, results: List[str]):
    """검색 결과를 캐시에 저장"""
    import threading
    global _pinecone_cache, _cache_lock
    
    if _cache_lock is None:
        _cache_lock = threading.Lock()
    
    cache_key = _get_cache_key(query, top_k)
    
    with _cache_lock:
        # 캐시 크기 제한 (LRU 방식)
        if len(_pinecone_cache) >= _cache_max_size:
            # 가장 오래된 항목 제거 (FIFO)
            oldest_key = next(iter(_pinecone_cache))
            del _pinecone_cache[oldest_key]
        
        _pinecone_cache[cache_key] = {
            'results': results,
            'top_k': top_k,
            'timestamp': datetime.now().isoformat()
        }


def _get_pipeline():
    """파이프라인 싱글톤 인스턴스 반환 (지연 초기화)"""
    global _pipeline_instance, _pipeline_lock
    
    if _pipeline_instance is None:
        import threading
        if _pipeline_lock is None:
            _pipeline_lock = threading.Lock()
        
        with _pipeline_lock:
            if _pipeline_instance is None:
                from app.services.pinecone_pipeline import PanelSearchPipeline
                category_config = load_category_config()
                
                if not PINECONE_API_KEY:
                    raise ValueError("PINECONE_API_KEY 환경변수가 설정되지 않았습니다.")
                
                _pipeline_instance = PanelSearchPipeline(
                    pinecone_api_key=PINECONE_API_KEY,
                    pinecone_index_name=PINECONE_INDEX_NAME,
                    category_config=category_config,
                    anthropic_api_key=ANTHROPIC_API_KEY,
                    openai_api_key=OPENAI_API_KEY
                )
                logger.info("Pinecone 파이프라인 초기화 완료")
    
    return _pipeline_instance


async def _search_with_pinecone(
    query_text: str,
    top_k: int = 100,
    use_cache: bool = True,
    force_refresh: bool = False,
    filters_dict: Optional[Dict[str, Any]] = None
) -> Optional[Dict[str, Any]]:
    """
    Pinecone을 사용한 패널 검색 (비동기 실행, 파이프라인 재사용, 캐싱 지원)
    
    Args:
        query_text: 검색 쿼리
        top_k: 반환할 패널 수
        use_cache: 캐시 사용 여부
        force_refresh: 강제 새로고침 (캐시 무시)
        
    Returns:
        {"mb_sns": [...], "scores": {...}} 또는 None (실패 시)
        호환성을 위해 List[str]도 반환 가능
    """
    if not PINECONE_SEARCH_ENABLED:
        return None
    
    # 캐시 확인 (강제 새로고침이 아니고 캐시 사용이 활성화된 경우)
    if use_cache and not force_refresh:
        cached_result = _get_cached_result(query_text, top_k)
        if cached_result is not None:
            return cached_result
    
    try:
        import asyncio
        import time
        
        # 싱글톤 파이프라인 가져오기
        pipeline = _get_pipeline()
        
        # 프론트엔드 필터를 Pinecone 필터로 변환
        external_filters = None
        if filters_dict:
            # ⭐ 빈 필터 체크: 실제로 값이 있는 필터만 있는지 확인
            has_actual_filters = any(
                (isinstance(v, list) and len(v) > 0) or
                (isinstance(v, bool) and v is True) or
                (isinstance(v, (int, float)) and v > 0) or
                (isinstance(v, str) and v.strip())
                for v in filters_dict.values()
            )
            
            if has_actual_filters:
                converter = PineconeFilterConverter()
                external_filters = converter.convert_to_pinecone_filters(filters_dict)
            else:
                external_filters = None
        
        # 동기 함수를 비동기로 실행 (LLM 호출 등 블로킹 작업 포함)
        # 타임아웃 설정: 240초 (LLM 호출이 여러 단계에서 발생하므로 여유있게 설정)
        executor_start_time = time.time()
        loop = asyncio.get_event_loop()
        try:
            search_result = await asyncio.wait_for(
                loop.run_in_executor(
            None, 
            lambda: pipeline.search(query_text, top_k=top_k, external_filters=external_filters)
                ),
                timeout=240.0  # 240초 타임아웃
            )
        except asyncio.TimeoutError:
            executor_time = time.time() - executor_start_time
            logger.error(f"[Pinecone 검색] pipeline.search 타임아웃: {executor_time:.2f}초 경과 (240초 초과)")
            raise Exception(f"검색이 타임아웃되었습니다 (240초 초과). 서버 로그를 확인하세요.")
        
        # 검색 성공 시 캐시에 저장 (호환성을 위해 mb_sn 리스트만 캐시)
        if search_result is not None and use_cache:
            mb_sns = search_result.get("mb_sns", []) if isinstance(search_result, dict) else search_result
            if mb_sns:
                _set_cached_result(query_text, top_k, mb_sns)
        
        return search_result
        
    except Exception as e:
        logger.error(f"Pinecone 검색 실패: {e}", exc_info=True)
        # 오류 발생 시에도 캐시된 결과가 있으면 반환 (fallback)
        if use_cache and not force_refresh:
            cached_result = _get_cached_result(query_text, top_k)
            if cached_result is not None:
                logger.warning(f"[Pinecone 캐시] 검색 실패, 캐시된 결과 반환: '{query_text}'")
                return cached_result
        return None


@router.post("/api/search")
async def api_search_post(
    payload: Dict[str, Any]
):
    """패널 검색 API - Pinecone 검색 사용, 필터 지원"""
    import time
    request_start_time = time.time()
    
    try:
        filters_dict = payload.get("filters") or {}
        query_text = payload.get("query") or filters_dict.get("query")
        page = int(payload.get("page", 1))
        limit = int(payload.get("limit", 20))  # 페이지네이션용 limit (기본값 20)
        
        # 쿼리가 있는 경우: Pinecone 검색
        if query_text and str(query_text).strip():
            if not PINECONE_SEARCH_ENABLED:
                logger.warning("Pinecone 검색이 비활성화되어 있습니다.")
                return {
                    "query": query_text,
                    "page": page,
                    "page_size": limit,
                    "count": 0,
                    "total": 0,
                    "pages": 0,
                    "mode": "pinecone",
                    "error": "Pinecone 검색이 비활성화되어 있습니다.",
                    "results": []
                }
            
            import time
            api_start = time.time()
            
            # 프론트엔드에서 요청한 limit 값은 페이지네이션용이므로 무시
            # 쿼리에서 인원수를 추출하거나, 없으면 기본값 10 사용 (명수 미설정 시)
            # 실제 검색 개수는 파이프라인에서 쿼리에서 추출한 인원수 또는 기본값 10 사용
            # 여기서는 None을 전달하여 파이프라인에서 처리하도록 함
            search_top_k = None  # 파이프라인에서 쿼리에서 인원수 추출 또는 기본값 10 사용
            
            force_refresh = payload.get("force_refresh", False)  # 재검색 버튼 클릭 시 True
            pinecone_start = time.time()
            pinecone_result = await _search_with_pinecone(
                query_text, 
                top_k=search_top_k,
                use_cache=True,
                force_refresh=force_refresh,
                filters_dict=filters_dict  # 필터 전달
            )
            # 결과 형식 확인 (기존 List[str] 또는 새로운 Dict 형태)
            if pinecone_result is None:
                pinecone_mb_sns = None
                pinecone_scores = {}
            elif isinstance(pinecone_result, dict):
                pinecone_mb_sns = pinecone_result.get("mb_sns", [])
                pinecone_scores = pinecone_result.get("scores", {})
            else:
                # 기존 형식 (List[str]) - 호환성 유지
                pinecone_mb_sns = pinecone_result
                pinecone_scores = {}
            
            if pinecone_mb_sns:
                # 필터링된 결과를 기존 API 형식으로 변환 (Pinecone 메타데이터 사용)
                panel_details = await _get_panel_details_from_pinecone(
                    pinecone_mb_sns, 1, len(pinecone_mb_sns), similarity_scores=pinecone_scores
                )
                
                response_data = {
                    "query": query_text,
                    "page": 1,
                    "page_size": len(panel_details["results"]),
                    "count": len(panel_details["results"]),
                    "total": len(panel_details["results"]),
                    "pages": 1,
                    "mode": "pinecone",
                    "results": panel_details["results"]
                }
                
                return response_data
            else:
                # 필터로 인해 결과가 0개
                return {
                    "query": query_text,
                    "page": page,
                    "page_size": limit,
                    "count": 0,
                    "total": 0,
                    "pages": 0,
                    "mode": "pinecone",
                    "error": "필터 조건에 맞는 결과가 없습니다.",
                    "results": []
                }
        
        # 쿼리가 없거나 Pinecone 검색 실패 시: 필터만으로 Pinecone 검색
        if filters_dict and any(filters_dict.values()):
            try:
                # 필터를 Pinecone 필터로 변환
                converter = PineconeFilterConverter()
                external_filters = converter.convert_to_pinecone_filters(filters_dict)
                
                if external_filters:
                    # 빈 쿼리로 Pinecone 검색 (필터만 적용)
                    # ⭐ 필터만 검색하는 경우 조건에 부합하는 모든 패널 반환 (top_k=None)
                    import asyncio
                    pipeline = _get_pipeline()
                    loop = asyncio.get_event_loop()
                    
                    # 빈 쿼리로 검색 (필터만 적용, 전체 반환)
                    search_result = await loop.run_in_executor(
                        None,
                        lambda: pipeline.search("", top_k=None, external_filters=external_filters)
                    )
                    
                    # pipeline.search()는 {"mb_sns": [...], "scores": {...}} 형태로 반환
                    if isinstance(search_result, dict):
                        mb_sn_list = search_result.get("mb_sns", [])
                        scores = search_result.get("scores", {})
                    elif isinstance(search_result, list):
                        # 호환성을 위해 리스트 형태도 처리
                        mb_sn_list = search_result
                        scores = {}
                    else:
                        mb_sn_list = []
                        scores = {}
                    
                    if mb_sn_list:
                        
                        # ⭐ 전체 mb_sn_list를 전달하고, _get_panel_details_from_pinecone 내부에서 페이지네이션 처리
                        # 전체 개수를 실제 검색 결과 개수로 설정
                        total_count = len(mb_sn_list)
                        import math
                        total_pages = math.ceil(total_count / limit) if limit > 0 else 1
                        
                        # 패널 상세 정보 조회 (전체 리스트 전달, 내부에서 페이지네이션)
                        panel_details = await _get_panel_details_from_pinecone(
                            mb_sn_list, page, limit, similarity_scores=scores
                        )
                        
                        # ⭐ 반환된 total을 전체 검색 결과 개수로 덮어쓰기
                        panel_details["total"] = total_count
                        panel_details["pages"] = total_pages
                        
                        return {
                            "query": "",
                            "page": page,
                            "page_size": limit,
                            "count": panel_details["count"],
                            "total": total_count,  # 전체 검색 결과 개수
                            "pages": total_pages,
                            "mode": "pinecone_filter",
                            "results": panel_details["results"]
                        }
            except Exception as e:
                logger.error(f"필터 검색 실패: {e}", exc_info=True)
        
        # Fallback: 빈 결과 반환
        return {
            "query": query_text or "",
            "page": page,
            "page_size": limit,
            "count": 0,
            "total": 0,
            "pages": 0,
            "mode": "pinecone",
            "error": "검색 결과가 없습니다.",
            "results": []
        }
    except Exception as e:
        import traceback
        import time
        error_detail = traceback.format_exc()
        request_duration = time.time() - request_start_time
        logger.error(f"[API] 검색 API 오류 발생: {type(e).__name__} - {str(e)} (처리 시간: {request_duration:.2f}초)")
        raise HTTPException(
            status_code=500,
            detail=f"검색 중 오류 발생: {str(e)}"
        )
