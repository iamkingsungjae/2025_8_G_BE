"""merged.panel_data 테이블 데이터 로더 (순환 import 방지)"""
from pathlib import Path
from typing import Dict, Any, Optional
import json
import logging
import asyncio
from sqlalchemy import text

logger = logging.getLogger(__name__)

# 프로젝트 루트 경로 (fallback용)
PROJECT_ROOT = Path(__file__).resolve().parents[3]
MERGED_FINAL_JSON = PROJECT_ROOT / 'merged_final.json'

# merged.panel_data 데이터를 메모리에 캐싱
_merged_data_cache: Optional[Dict[str, Any]] = None
_cache_lock = asyncio.Lock()


async def load_merged_data_from_db() -> Dict[str, Any]:
    """merged.panel_data 테이블에서 데이터를 로드하고 mb_sn을 키로 하는 딕셔너리로 변환
    
    Returns:
        mb_sn을 키로 하는 딕셔너리
    """
    global _merged_data_cache
    
    # 캐시 확인 (동기적으로)
    if _merged_data_cache is not None:
        logger.info(f"[Merged Data] 캐시된 merged_data 사용: {len(_merged_data_cache)}개 패널")
        return _merged_data_cache
    
    # 비동기 락으로 중복 로드 방지
    async with _cache_lock:
        # 다시 확인 (락 획득 후)
        if _merged_data_cache is not None:
            return _merged_data_cache
        
        try:
            # 환경변수를 직접 읽어서 엔진 생성 (모듈 로드 시점의 engine 사용 방지)
            import os
            from dotenv import load_dotenv
            from sqlalchemy.ext.asyncio import create_async_engine
            import sys
            
            load_dotenv(override=True)
            
            uri = os.getenv("ASYNC_DATABASE_URI")
            if not uri:
                logger.error("[Merged Data] ASYNC_DATABASE_URI 환경변수가 설정되지 않았습니다.")
                return _load_merged_data_from_json_fallback()
            
            # postgresql://를 postgresql+psycopg://로 변환
            if uri.startswith("postgresql://"):
                uri = uri.replace("postgresql://", "postgresql+psycopg://", 1)
            elif "postgresql+asyncpg" in uri:
                uri = uri.replace("postgresql+asyncpg", "postgresql+psycopg", 1)
            
            # Windows 이벤트 루프 정책 설정
            if sys.platform == 'win32':
                asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
            
            # 새 엔진 생성
            temp_engine = create_async_engine(uri, echo=False, pool_pre_ping=True, poolclass=None)
            
            logger.info(f"[Merged Data] merged.panel_data 테이블에서 로드 시작...")
            
            async with temp_engine.begin() as conn:
                # merged 스키마로 search_path 설정
                await conn.execute(text('SET search_path TO "merged", public'))
                
                # merged.panel_data 테이블에서 모든 데이터 조회
                result = await conn.execute(text("""
                    SELECT * FROM merged.panel_data
                """))
                
                rows = result.mappings().all()
                logger.info(f"[Merged Data] DB에서 {len(rows)}개 행 조회 완료")
                
                # mb_sn을 키로 하는 딕셔너리로 변환
                # base_profile JSONB를 파싱하여 평탄화
                converted_data = {}
                for row in rows:
                    row_dict = dict(row)
                    mb_sn = row_dict.get('mb_sn')
                    if not mb_sn:
                        continue
                    
                    # base_profile JSONB 파싱 (PostgreSQL에서 자동으로 dict로 변환됨)
                    base_profile = row_dict.get('base_profile', {})
                    if not isinstance(base_profile, dict):
                        # JSONB가 문자열로 반환된 경우 파싱
                        import json
                        if isinstance(base_profile, str):
                            base_profile = json.loads(base_profile)
                        else:
                            base_profile = {}
                    
                    quick_answers = row_dict.get('quick_answers', {})
                    if not isinstance(quick_answers, dict):
                        import json
                        if isinstance(quick_answers, str):
                            quick_answers = json.loads(quick_answers)
                        else:
                            quick_answers = {}
                    
                    # base_profile의 모든 필드를 평탄화하여 저장
                    panel_data = {
                        'mb_sn': mb_sn,
                        **base_profile  # base_profile의 모든 필드를 펼침
                    }
                    
                    # quick_answers가 있으면 추가
                    if quick_answers:
                        panel_data['quick_answers'] = quick_answers
                    
                    converted_data[mb_sn] = panel_data
                
                _merged_data_cache = converted_data
                logger.info(f"[Merged Data] 딕셔너리 변환 완료: {len(_merged_data_cache)}개 패널")
                return _merged_data_cache
                
            await temp_engine.dispose()
                
        except Exception as e:
            logger.error(f"[ERROR] merged.panel_data 로드 실패: {str(e)}", exc_info=True)
            # Fallback: JSON 파일 시도
            logger.warning(
                f"[Merged Data] ⚠️ NeonDB 로드 실패, JSON 파일 fallback 시도\n"
                f"  → 모든 패널 데이터는 NeonDB의 merged.panel_data 테이블에 저장되어야 합니다.\n"
                f"  → JSON fallback은 개발/테스트 목적으로만 사용됩니다."
            )
            return _load_merged_data_from_json_fallback()


def _load_merged_data_from_json_fallback() -> Dict[str, Any]:
    """
    JSON 파일에서 로드 (fallback)
    
    ⚠️ 주의: 이 함수는 개발/테스트 목적으로만 사용됩니다.
    프로덕션 환경에서는 모든 데이터가 NeonDB에 저장되어야 하며,
    이 fallback은 사용되지 않아야 합니다.
    """
    global _merged_data_cache
    
    logger.warning(
        f"[Merged Data] ⚠️ DB 로드 실패, JSON 파일로 fallback 시도: {MERGED_FINAL_JSON}\n"
        f"  → 프로덕션 환경에서는 이 fallback이 사용되지 않아야 합니다.\n"
        f"  → 모든 패널 데이터는 NeonDB의 merged.panel_data 테이블에 저장되어야 합니다."
    )
    if not MERGED_FINAL_JSON.exists():
        logger.warning(f"[Merged Data] 경고: merged_final.json 파일이 존재하지 않음: {MERGED_FINAL_JSON}")
        return {}
    
    try:
        logger.info(f"[Merged Data] JSON 파일 읽기 시작...")
        with open(MERGED_FINAL_JSON, 'r', encoding='utf-8') as f:
            data = json.load(f)
        logger.info(f"[Merged Data] JSON 파일 읽기 완료: {len(data)}개 항목")
        
        converted_data = {}
        for item in data:
            if 'mb_sn' not in item:
                continue
            
            mb_sn = item['mb_sn']
            converted_data[mb_sn] = item
        
        _merged_data_cache = converted_data
        logger.info(f"[Merged Data] 딕셔너리 변환 완료: {len(_merged_data_cache)}개 패널")
        return _merged_data_cache
    except Exception as e:
        logger.error(f"[ERROR] merged_final.json 로드 실패: {str(e)}", exc_info=True)
        return {}


def load_merged_data() -> Dict[str, Any]:
    """merged.panel_data 테이블에서 데이터를 로드 (동기 인터페이스)
    
    비동기 함수를 동기적으로 호출합니다.
    기존 코드와의 호환성을 위해 동기 인터페이스를 유지합니다.
    
    Returns:
        mb_sn을 키로 하는 딕셔너리
    """
    global _merged_data_cache
    
    # 캐시 확인
    if _merged_data_cache is not None:
        logger.info(f"[Merged Data] 캐시된 merged_data 사용: {len(_merged_data_cache)}개 패널")
        return _merged_data_cache
    
    try:
        # 비동기 함수를 동기적으로 실행
        import asyncio
        import sys
        
        # Windows 이벤트 루프 정책 설정
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
        # 이벤트 루프 상태 확인
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 이미 실행 중인 루프에서는 새 스레드에서 실행
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, load_merged_data_from_db())
                    _merged_data_cache = future.result(timeout=60)
            else:
                _merged_data_cache = loop.run_until_complete(load_merged_data_from_db())
        except RuntimeError:
            # 이벤트 루프가 없으면 새로 생성
            _merged_data_cache = asyncio.run(load_merged_data_from_db())
        
        logger.info(f"[Merged Data] DB에서 로드 완료: {len(_merged_data_cache)}개 패널")
        return _merged_data_cache
        
    except Exception as e:
        logger.error(f"[ERROR] merged.panel_data 로드 실패: {str(e)}", exc_info=True)
        logger.warning(
            f"[Merged Data] ⚠️ NeonDB 로드 실패, JSON 파일 fallback 시도\n"
            f"  → 모든 패널 데이터는 NeonDB의 merged.panel_data 테이블에 저장되어야 합니다.\n"
            f"  → JSON fallback은 개발/테스트 목적으로만 사용됩니다."
        )
        return _load_merged_data_from_json_fallback()


async def get_panels_from_merged_db_batch(panel_ids: list[str]) -> Dict[str, Dict[str, Any]]:
    """merged.panel_data 테이블에서 여러 패널 데이터를 한 번에 조회
    
    Args:
        panel_ids: 패널 ID 리스트 (mb_sn)
        
    Returns:
        mb_sn을 키로 하는 딕셔너리 (패널 ID -> 패널 데이터)
    """
    logger.info(f"[Merged Data] 배치 패널 조회 시작: {len(panel_ids)}개")
    try:
        # 먼저 캐시에서 확인
        global _merged_data_cache
        if _merged_data_cache is not None:
            result = {}
            for panel_id in panel_ids:
                panel_data = _merged_data_cache.get(panel_id)
                if panel_data:
                    result[panel_id] = panel_data
            if len(result) == len(panel_ids):
                logger.info(f"[Merged Data] 캐시에서 모든 패널 조회 성공: {len(result)}개")
                return result
            else:
                logger.info(f"[Merged Data] 캐시에서 일부 패널 조회: {len(result)}/{len(panel_ids)}개, 나머지는 DB에서 조회")
        
        # 캐시가 없거나 일부만 있으면 DB에서 직접 조회
        import os
        from dotenv import load_dotenv
        from sqlalchemy.ext.asyncio import create_async_engine
        import sys
        
        load_dotenv(override=True)
        
        uri = os.getenv("ASYNC_DATABASE_URI")
        if not uri:
            logger.error("[Merged Data] ASYNC_DATABASE_URI 환경변수가 설정되지 않았습니다.")
            return {}
        
        # postgresql://를 postgresql+psycopg://로 변환
        if uri.startswith("postgresql://"):
            uri = uri.replace("postgresql://", "postgresql+psycopg://", 1)
        elif "postgresql+asyncpg" in uri:
            uri = uri.replace("postgresql+asyncpg", "postgresql+psycopg", 1)
        
        # Windows 이벤트 루프 정책 설정
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
        # 새 엔진 생성
        temp_engine = create_async_engine(uri, echo=False, pool_pre_ping=True, poolclass=None)
        
        logger.info(f"[Merged Data] DB에서 배치 패널 조회 시작: {len(panel_ids)}개")
        
        try:
            async with temp_engine.begin() as conn:
                # merged 스키마로 search_path 설정
                await conn.execute(text('SET search_path TO "merged", public'))
                
                # 여러 패널을 한 번에 조회 (IN 절 사용)
                result = await conn.execute(
                    text("SELECT * FROM merged.panel_data WHERE mb_sn = ANY(:mb_sns)"),
                    {"mb_sns": panel_ids}
                )
                
                rows = result.mappings().all()
                logger.info(f"[Merged Data] DB에서 {len(rows)}개 행 조회 완료")
                
                # mb_sn을 키로 하는 딕셔너리로 변환
                result_dict = {}
                for row in rows:
                    row_dict = dict(row)
                    mb_sn = row_dict.get('mb_sn')
                    if not mb_sn:
                        continue
                    
                    # base_profile JSONB 파싱
                    base_profile = row_dict.get('base_profile', {})
                    if not isinstance(base_profile, dict):
                        import json
                        if isinstance(base_profile, str):
                            base_profile = json.loads(base_profile)
                        else:
                            base_profile = {}
                    
                    quick_answers = row_dict.get('quick_answers', {})
                    if not isinstance(quick_answers, dict):
                        import json
                        if isinstance(quick_answers, str):
                            quick_answers = json.loads(quick_answers)
                        else:
                            quick_answers = {}
                    
                    # base_profile의 모든 필드를 평탄화하여 저장
                    panel_data = {
                        'mb_sn': mb_sn,
                        **base_profile  # base_profile의 모든 필드를 펼침
                    }
                    
                    # quick_answers가 있으면 추가
                    if quick_answers:
                        panel_data['quick_answers'] = quick_answers
                    
                    result_dict[mb_sn] = panel_data
                
                logger.info(f"[Merged Data] DB에서 배치 패널 조회 완료: {len(result_dict)}개")
                await temp_engine.dispose()
                return result_dict
                
        except Exception as db_error:
            logger.error(f"[Merged Data] DB 배치 조회 중 오류 발생: {str(db_error)}", exc_info=True)
            try:
                await temp_engine.dispose()
            except:
                pass
            raise
            
    except Exception as e:
        logger.error(f"[ERROR] merged.panel_data에서 배치 패널 조회 실패: {str(e)}", exc_info=True)
        return {}


async def get_panel_from_merged_db(panel_id: str) -> Optional[Dict[str, Any]]:
    """merged.panel_data 테이블에서 특정 패널 데이터 조회
    
    Args:
        panel_id: 패널 ID (mb_sn)
        
    Returns:
        패널 데이터 딕셔너리 또는 None
    """
    logger.info(f"[Merged Data] 패널 조회 시작: {panel_id}")
    try:
        # 먼저 캐시에서 확인
        global _merged_data_cache
        if _merged_data_cache is not None:
            panel_data = _merged_data_cache.get(panel_id)
            if panel_data:
                logger.info(f"[Merged Data] 캐시에서 패널 조회 성공: {panel_id}")
                return panel_data
            else:
                logger.info(f"[Merged Data] 캐시에 패널 없음: {panel_id}, DB에서 조회 시도")
        
        # 캐시가 없거나 패널이 없으면 DB에서 직접 조회
        import os
        from dotenv import load_dotenv
        from sqlalchemy.ext.asyncio import create_async_engine
        import sys
        
        load_dotenv(override=True)
        
        uri = os.getenv("ASYNC_DATABASE_URI")
        if not uri:
            logger.error("[Merged Data] ASYNC_DATABASE_URI 환경변수가 설정되지 않았습니다.")
            return None
        
        # postgresql://를 postgresql+psycopg://로 변환
        if uri.startswith("postgresql://"):
            uri = uri.replace("postgresql://", "postgresql+psycopg://", 1)
        elif "postgresql+asyncpg" in uri:
            uri = uri.replace("postgresql+asyncpg", "postgresql+psycopg", 1)
        
        # Windows 이벤트 루프 정책 설정
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
        # 새 엔진 생성
        temp_engine = create_async_engine(uri, echo=False, pool_pre_ping=True, poolclass=None)
        
        logger.info(f"[Merged Data] DB에서 패널 조회 시작: {panel_id}")
        
        try:
            async with temp_engine.begin() as conn:
                # merged 스키마로 search_path 설정
                await conn.execute(text('SET search_path TO "merged", public'))
                
                # 특정 패널 조회
                result = await conn.execute(
                    text("SELECT * FROM merged.panel_data WHERE mb_sn = :mb_sn"),
                    {"mb_sn": panel_id}
                )
                
                row = result.mappings().first()
                
                if not row:
                    logger.warning(f"[Merged Data] 패널을 찾을 수 없음: {panel_id}")
                    await temp_engine.dispose()
                    return None
                
                row_dict = dict(row)
                mb_sn = row_dict.get('mb_sn')
                if not mb_sn:
                    logger.warning(f"[Merged Data] 패널 데이터에 mb_sn이 없음: {panel_id}")
                    await temp_engine.dispose()
                    return None
                
                # base_profile JSONB 파싱
                base_profile = row_dict.get('base_profile', {})
                if not isinstance(base_profile, dict):
                    import json
                    if isinstance(base_profile, str):
                        base_profile = json.loads(base_profile)
                    else:
                        base_profile = {}
                
                quick_answers = row_dict.get('quick_answers', {})
                if not isinstance(quick_answers, dict):
                    import json
                    if isinstance(quick_answers, str):
                        quick_answers = json.loads(quick_answers)
                    else:
                        quick_answers = {}
                
                # base_profile의 모든 필드를 평탄화하여 저장
                panel_data = {
                    'mb_sn': mb_sn,
                    **base_profile  # base_profile의 모든 필드를 펼침
                }
                
                # quick_answers가 있으면 추가
                if quick_answers:
                    panel_data['quick_answers'] = quick_answers
                
                logger.info(f"[Merged Data] DB에서 패널 조회 완료: {panel_id}")
                await temp_engine.dispose()
                return panel_data
                
        except Exception as db_error:
            logger.error(f"[Merged Data] DB 조회 중 오류 발생: {panel_id}, 오류: {str(db_error)}", exc_info=True)
            try:
                await temp_engine.dispose()
            except:
                pass
            # 에러를 다시 발생시켜 상위에서 처리하도록
            raise
            
    except Exception as e:
        logger.error(f"[ERROR] merged.panel_data에서 패널 조회 실패: {panel_id}, 오류: {str(e)}", exc_info=True)
        return None


def get_panel_from_merged(panel_id: str) -> Optional[Dict[str, Any]]:
    """merged.panel_data 테이블에서 특정 패널 데이터 조회 (동기 인터페이스)
    
    Args:
        panel_id: 패널 ID (mb_sn)
        
    Returns:
        패널 데이터 딕셔너리 또는 None
    """
    # 먼저 캐시에서 확인
    global _merged_data_cache
    if _merged_data_cache is not None:
        panel_data = _merged_data_cache.get(panel_id)
        if panel_data:
            logger.info(f"[Merged Data] 캐시에서 패널 조회: {panel_id}")
            return panel_data
    
    # 캐시가 없으면 비동기 함수를 동기적으로 실행
    try:
        import asyncio
        import sys
        
        # Windows 이벤트 루프 정책 설정
        if sys.platform == 'win32':
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
        # 이벤트 루프 상태 확인
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # 이미 실행 중인 루프에서는 새 스레드에서 실행
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(asyncio.run, get_panel_from_merged_db(panel_id))
                    return future.result(timeout=30)
            else:
                return loop.run_until_complete(get_panel_from_merged_db(panel_id))
        except RuntimeError:
            # 이벤트 루프가 없으면 새로 생성
            return asyncio.run(get_panel_from_merged_db(panel_id))
        
    except Exception as e:
        logger.error(f"[ERROR] merged.panel_data에서 패널 조회 실패: {panel_id}, 오류: {str(e)}", exc_info=True)
        return None

