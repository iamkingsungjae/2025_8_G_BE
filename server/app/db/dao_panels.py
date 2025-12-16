"""패널 검색 공통 DAO 및 쿼리 빌더"""
from typing import Any, Dict, List, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from app.core.config import DBN, fq


def build_search_sql(filters: Dict[str, Any]) -> tuple[str, Dict[str, Any]]:
    """
    패널 검색 SQL 쿼리 빌더 (필터 + 텍스트 검색)
    
    Args:
        filters: 검색 필터 딕셔너리
            - query: 텍스트 검색어 (RawData의 모든 텍스트 필드에서 LIKE 검색)
            - gender: 성별 ('M'/'F'/'남'/'여' 모두 허용) 또는 리스트
            - region: 지역 또는 리스트
            - age_min: 최소 나이
            - age_max: 최대 나이
            - limit: 페이지 크기 (기본 50)
            - offset: 오프셋 (기본 0)
            
    Returns:
        (SQL 쿼리 문자열, 파라미터 딕셔너리) 튜플
    """
    W1 = fq(DBN.RAW, DBN.T_W1)
    W2 = fq(DBN.RAW, DBN.T_W2)
    QA = fq(DBN.RAW, DBN.T_QA)
    
    sql = f"""
    SELECT
      w1.mb_sn,
      w1.gender,
      CASE 
        WHEN COALESCE(NULLIF(w1.birth_year, ''), NULL) IS NOT NULL
             AND w1.birth_year ~ '^[0-9]+$'
        THEN EXTRACT(YEAR FROM AGE(CURRENT_DATE, MAKE_DATE(w1.birth_year::int, 1, 1)))::int
        ELSE NULL 
      END AS age_raw,
      w1."location" AS location,
      w1.detail_location,
      w2."data" AS w2_data,
      qa.answers AS qa_answers
    FROM {W1} w1
    LEFT JOIN {W2} w2 ON w1.mb_sn = w2.mb_sn
    LEFT JOIN {QA} qa ON w1.mb_sn = qa.mb_sn
    WHERE 1=1
    """
    
    params: Dict[str, Any] = {}
    
    # 텍스트 검색 (query 파라미터)
    if query_text := filters.get("query"):
        if str(query_text).strip():  # 빈 문자열 체크
            # RawData의 모든 텍스트 필드에서 검색
            # w2_data (JSONB), qa_answers (JSONB), location, detail_location 등
            # NULL 안전하게 처리
            sql += """ AND (
                LOWER(COALESCE(w1."location", '') || ' ' || COALESCE(w1.detail_location, '')) LIKE :q_text
                OR LOWER(COALESCE(w2."data"::text, '')) LIKE :q_text
                OR LOWER(COALESCE(qa.answers::text, '')) LIKE :q_text
            ) """
            params["q_text"] = f"%{str(query_text).strip().lower()}%"
    
    # 성별 필터 (리스트 지원)
    if g := filters.get("gender"):
        if isinstance(g, list):
            # 리스트인 경우: IN 절 사용
            gender_conditions = []
            for idx, gender_val in enumerate(g):
                g_lower = str(gender_val).lower()
                gender_map = {"m": "남성", "f": "여성", "male": "남성", "female": "여성", "남": "남성", "여": "여성"}
                g_normalized = gender_map.get(g_lower, g_lower)
                gender_conditions.append(f"(LOWER(COALESCE(w1.gender, '')) = :g{idx}_1 OR LOWER(COALESCE(w1.gender, '')) = :g{idx}_2)")
                params[f"g{idx}_1"] = g_normalized.lower()
                params[f"g{idx}_2"] = g_lower
            if gender_conditions:
                sql += " AND (" + " OR ".join(gender_conditions) + ") "
        else:
            # 단일 값인 경우
            g_lower = str(g).lower()
            gender_map = {"m": "남성", "f": "여성", "male": "남성", "female": "여성", "남": "남성", "여": "여성"}
            g_normalized = gender_map.get(g_lower, g_lower)
            sql += " AND (LOWER(COALESCE(w1.gender, '')) = :g1 OR LOWER(COALESCE(w1.gender, '')) = :g2) "
            params["g1"] = g_normalized.lower()
            params["g2"] = g_lower
    
    # 지역 필터 (리스트 지원, 정확 매칭)
    if r := filters.get("region"):
        if isinstance(r, list):
            # 리스트인 경우: IN 절 사용
            region_conditions = []
            for idx, region_val in enumerate(r):
                region_conditions.append(f"LOWER(COALESCE(w1.\"location\", '')) = :r{idx}")
                params[f"r{idx}"] = str(region_val).lower()
            if region_conditions:
                sql += " AND (" + " OR ".join(region_conditions) + ") "
        else:
            # 단일 값인 경우: 정확 매칭
            sql += " AND LOWER(COALESCE(w1.\"location\", '')) = :r "
            params["r"] = str(r).lower()
    
    # 나이 필터
    if age_min := filters.get("age_min"):
        sql += " AND (EXTRACT(YEAR FROM AGE(CURRENT_DATE, MAKE_DATE(w1.birth_year::int, 1, 1)))::int >= :age_min OR w1.birth_year IS NULL) "
        params["age_min"] = int(age_min)
    
    if age_max := filters.get("age_max"):
        sql += " AND (EXTRACT(YEAR FROM AGE(CURRENT_DATE, MAKE_DATE(w1.birth_year::int, 1, 1)))::int <= :age_max OR w1.birth_year IS NULL) "
        params["age_max"] = int(age_max)
    
    # 정렬 및 페이지네이션 (최대 200개로 제한)
    sql += " ORDER BY w1.mb_sn LIMIT :limit OFFSET :offset "
    params["limit"] = min(int(filters.get("limit", 200)), 200)  # 최대 200개
    params["offset"] = int(filters.get("offset", 0))
    
    return sql, params


async def search_panels(session: AsyncSession, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    패널 검색 실행
    
    Args:
        session: 비동기 데이터베이스 세션
        filters: 검색 필터 딕셔너리
        
    Returns:
        검색 결과 리스트 (딕셔너리 리스트)
    """
    try:
        # 안전을 위해 search_path 고정
        await session.execute(text(f'SET search_path TO "{DBN.RAW}", public'))
        
        # SQL 쿼리 빌드 및 실행
        sql, params = build_search_sql(filters)
        result = await session.execute(text(sql), params)
        rows = [dict(row) for row in result.mappings()]
        
        return rows
        
    except Exception as e:
        # 에러만 로깅
        import logging
        logger = logging.getLogger(__name__)
        logger.error(f"search_panels 오류: {str(e)}", exc_info=True)
        raise


async def get_panel_detail(session: AsyncSession, panel_id: str) -> Optional[Dict[str, Any]]:
    """
    패널 상세 정보 조회 (panel_embeddings_v 뷰 우선, 없으면 welcome_1st/welcome_2nd/quick_answer 직접 조회)
    
    Args:
        session: 비동기 데이터베이스 세션
        panel_id: 패널 ID (mb_sn)
        
    Returns:
        패널 상세 정보 딕셔너리 또는 None
    """
    EMB_V = fq(DBN.RAW, DBN.T_EMB_V)
    W1 = fq(DBN.RAW, DBN.T_W1)
    W2 = fq(DBN.RAW, DBN.T_W2)
    QA = fq(DBN.RAW, DBN.T_QA)
    
    # search_path 고정
    await session.execute(text(f'SET search_path TO "{DBN.RAW}", public'))
    
    # 1. panel_embeddings_v 뷰에서 먼저 조회 시도
    try:
        sql = f"""
        SELECT 
          emb.mb_sn,
          emb.demographics,
          emb.combined_text,
          emb.labeled_text,
          emb.chunks,
          emb.chunk_count,
          emb.categories,
          emb.all_labels,
          emb.created_at,
          -- welcome_1st 데이터
          w1.gender AS w1_gender,
          w1.birth_year AS w1_birth_year,
          w1."location" AS w1_location,
          w1.detail_location AS w1_detail_location,
          -- welcome_2nd 데이터
          w2."data" AS w2_data,
          -- quick_answer 데이터
          qa.answers AS qa_answers
        FROM {EMB_V} emb
        LEFT JOIN {W1} w1 ON emb.mb_sn = w1.mb_sn
        LEFT JOIN {W2} w2 ON emb.mb_sn = w2.mb_sn
        LEFT JOIN {QA} qa ON emb.mb_sn = qa.mb_sn
        WHERE emb.mb_sn = :mb_sn
        LIMIT 1
        """
        
        result = await session.execute(text(sql), {"mb_sn": panel_id})
        row = result.mappings().first()
        
        if row:
            return dict(row)
    except Exception as e:
        # 뷰가 없거나 오류가 발생하면 fallback으로 진행
        print(f"[DEBUG Panel Detail] panel_embeddings_v 뷰 조회 실패: {str(e)}, fallback으로 진행")
    
    # 2. Fallback: welcome_1st, welcome_2nd, quick_answer 테이블에서 직접 조회
    sql_fallback = f"""
    SELECT 
      w1.mb_sn,
      NULL::jsonb AS demographics,
      NULL::text AS combined_text,
      NULL::text AS labeled_text,
      NULL::jsonb AS chunks,
      NULL::integer AS chunk_count,
      NULL::jsonb AS categories,
      NULL::jsonb AS all_labels,
      NULL::timestamp AS created_at,
      -- welcome_1st 데이터
      w1.gender AS w1_gender,
      w1.birth_year AS w1_birth_year,
      w1."location" AS w1_location,
      w1.detail_location AS w1_detail_location,
      -- welcome_2nd 데이터
      w2."data" AS w2_data,
      -- quick_answer 데이터
      qa.answers AS qa_answers
    FROM {W1} w1
    LEFT JOIN {W2} w2 ON w1.mb_sn = w2.mb_sn
    LEFT JOIN {QA} qa ON w1.mb_sn = qa.mb_sn
    WHERE w1.mb_sn = :mb_sn
    LIMIT 1
    """
    
    result = await session.execute(text(sql_fallback), {"mb_sn": panel_id})
    row = result.mappings().first()
    
    if row:
        return dict(row)
    
    # 3. 최종 Fallback: 더 이상 사용할 수 있는 데이터 소스가 없음
    return None


async def count_panels(session: AsyncSession, filters: Dict[str, Any]) -> int:
    """
    패널 검색 결과 개수 조회
    
    Args:
        session: 비동기 데이터베이스 세션
        filters: 검색 필터 딕셔너리
        
    Returns:
        검색 결과 개수
    """
    import re
    
    await session.execute(text(f'SET search_path TO "{DBN.RAW}", public'))
    
    # COUNT 쿼리 (LIMIT/OFFSET 제외)
    count_filters = {k: v for k, v in filters.items() if k not in ("limit", "offset")}
    sql, params = build_search_sql(count_filters)
    
    # 서브쿼리 방식: ORDER BY/LIMIT/OFFSET 제거 후 서브쿼리로 감싸기
    # ORDER BY부터 끝까지 제거
    where_clause_end = sql.upper().rfind('WHERE')
    if where_clause_end == -1:
        # WHERE가 없으면 FROM 다음부터 찾기
        from_pos = sql.upper().find('FROM')
        if from_pos != -1:
            # FROM 다음부터 ORDER BY 전까지
            order_by_pos = sql.upper().find('ORDER BY', from_pos)
            if order_by_pos != -1:
                base_sql = sql[:order_by_pos].strip()
            else:
                base_sql = sql
        else:
            base_sql = sql
    else:
        # ORDER BY 찾기
        order_by_pos = sql.upper().find('ORDER BY', where_clause_end)
        if order_by_pos != -1:
            base_sql = sql[:order_by_pos].strip()
        else:
            base_sql = sql.strip()
    
    # SELECT 부분을 DISTINCT mb_sn으로 변경 (중복 제거를 위해)
    # SELECT ... FROM 을 SELECT DISTINCT w1.mb_sn FROM 로 변경
    base_sql = re.sub(
        r'SELECT\s+.*?\s+FROM',
        'SELECT DISTINCT w1.mb_sn FROM',
        base_sql,
        flags=re.DOTALL | re.IGNORECASE
    )
    
    # 서브쿼리로 감싸서 COUNT
    count_sql = f"SELECT COUNT(*) as count FROM ({base_sql}) sub"
    
    # limit, offset 파라미터 제거
    params_clean = {k: v for k, v in params.items() if k not in ("limit", "offset")}
    
    try:
        result = await session.execute(text(count_sql), params_clean)
        count = result.scalar() or 0
        return count
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"count_panels 오류: {str(e)}", exc_info=True)
        raise


def _sql_extract_features(panel_ids: List[str]) -> tuple[str, Dict[str, Any]]:
    """
    클러스터링용 피처 추출 SQL 쿼리 생성
    
    Args:
        panel_ids: 패널 ID 리스트
        
    Returns:
        (SQL 쿼리 문자열, 파라미터 딕셔너리) 튜플
    """
    W1, W2, QA = fq(DBN.RAW, DBN.T_W1), fq(DBN.RAW, DBN.T_W2), fq(DBN.RAW, DBN.T_QA)
    
    sql = f"""
    SELECT
      w1.mb_sn AS panel_id,
      -- 숫자 피처
      CASE 
        WHEN w1.birth_year ~ '^[0-9]{{4}}$'
        THEN (EXTRACT(YEAR FROM CURRENT_DATE)::int - w1.birth_year::int)
        ELSE NULL 
      END AS age_raw,
      NULLIF((w2."data"->>'income_personal')::numeric, NULL) AS income_personal,
      NULLIF((w2."data"->>'income_household')::numeric, NULL) AS income_household,
      -- 범주 피처
      CASE
        WHEN lower(w1.gender) IN ('m','남','남성','male') THEN 'M'
        WHEN lower(w1.gender) IN ('f','여','여성','female') THEN 'F'
        ELSE 'UNK' 
      END AS gender,
      COALESCE(w1."location", 'UNK') AS region_lvl1,
      COALESCE(w1.detail_location, 'UNK') AS region_lvl2,
      -- 텍스트 데이터 (Python 후처리용)
      COALESCE(w2."data"::text, '') AS data_text,
      COALESCE(qa.answers::text, '') AS answers_text,
      -- chunk 근사 (정확 값은 Python에서 재계산)
      (SELECT COUNT(*) FROM jsonb_object_keys(qa.answers)) AS chunk_hint
    FROM {W1} w1
    LEFT JOIN {W2} w2 ON w1.mb_sn = w2.mb_sn
    LEFT JOIN {QA} qa ON w1.mb_sn = qa.mb_sn
    WHERE w1.mb_sn = ANY(:panel_ids)
    """
    
    return sql, {"panel_ids": panel_ids}


async def extract_features_for_clustering(
    session: AsyncSession, 
    panel_ids: List[str]
) -> List[Dict[str, Any]]:
    """
    RawData에서 클러스터링용 피처 추출
    
    Args:
        session: 비동기 데이터베이스 세션
        panel_ids: 패널 ID 리스트
        
    Returns:
        피처 딕셔너리 리스트
    """
    # search_path 고정
    await session.execute(text(f'SET search_path TO "{DBN.RAW}", public'))
    
    sql, params = _sql_extract_features(panel_ids)
    result = await session.execute(text(sql), params)
    
    return [dict(row) for row in result.mappings()]


async def fetch_raw_sample(session: AsyncSession, limit: int = 10) -> List[Dict[str, Any]]:
    """
    RawData 스키마에서 기본 3테이블을 조인해 샘플 행을 반환.
    
    컬럼 존재 여부만 빠르게 확인하려면 한 테이블만 조회해도 OK.
    
    Args:
        session: 비동기 데이터베이스 세션
        limit: 반환할 행 수 (기본 10)
        
    Returns:
        샘플 행 딕셔너리 리스트
    """
    # search_path가 이미 "RawData"로 설정되어 있으므로 스키마 접두사 불필요
    # 하지만 대소문자 보장을 위해 따옴표 유지
    sql = text("""
        SELECT
          w1.mb_sn,
          w1.gender,
          CASE 
            WHEN w1.birth_year ~ '^[0-9]{4}$'
            THEN (EXTRACT(YEAR FROM CURRENT_DATE)::int - w1.birth_year::int)
            ELSE NULL 
          END AS age_raw,
          w1.location AS region,
          w2.data AS w2_data,
          qa.answers AS qa_answers
        FROM "welcome_1st" w1
        LEFT JOIN "welcome_2nd" w2 ON w1.mb_sn = w2.mb_sn
        LEFT JOIN "quick_answer" qa ON w1.mb_sn = qa.mb_sn
        ORDER BY w1.mb_sn
        LIMIT :limit
    """)
    
    # search_path 고정 (이미 설정되어 있지만 안전을 위해)
    await session.execute(text(f'SET search_path TO "{DBN.RAW}", public'))
    
    result = await session.execute(sql, {"limit": limit})
    return [dict(row) for row in result.mappings()]

