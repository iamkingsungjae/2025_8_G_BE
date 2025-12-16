"""임베딩 벡터 검색 DAO"""
import logging
from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from app.core.config import DBN, fq


async def find_embedding_tables(session: AsyncSession) -> List[Dict[str, Any]]:
    """
    임베딩 테이블 찾기 (모든 스키마에서 검색)
    
    Returns:
        발견된 테이블 리스트
    """
    
    try:
        # 모든 스키마에서 embedding 컬럼이 있는 테이블 검색
        search_sql = """
        SELECT 
            table_schema,
            table_name,
            column_name,
            data_type,
            udt_name
        FROM information_schema.columns 
        WHERE column_name = 'embedding'
        AND (udt_name = 'vector' OR data_type LIKE '%vector%')
        ORDER BY table_schema, table_name;
        """
        
        result = await session.execute(text(search_sql))
        tables = []
        for row in result:
            tables.append({
                "schema": row[0],
                "table": row[1],
                "column": row[2],
                "data_type": row[3],
                "udt_name": row[4]
            })
        
        return tables
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return []


async def create_panel_embeddings_view(session: AsyncSession, source_schema: str = None, source_table: str = None) -> Dict[str, Any]:
    """
    panel_embeddings_v 뷰 생성
    
    Args:
        session: 비동기 데이터베이스 세션
        source_schema: 원본 스키마 (없으면 자동 검색)
        source_table: 원본 테이블 (없으면 자동 검색)
    
    Returns:
        생성 결과 딕셔너리
    """
    
    try:
        # 1. 원본 테이블 찾기
        if not source_schema or not source_table:
            tables = await find_embedding_tables(session)
            
            if not tables:
                return {
                    "created": False,
                    "error": "임베딩 테이블을 찾을 수 없습니다. embedding 컬럼이 있는 테이블이 없습니다.",
                    "table_exists": False,
                    "searched_tables": []
                }
            
            # testcl.panel_embeddings 우선 선택, 없으면 첫 번째 테이블 사용
            testcl_table = next(
                (t for t in tables if t['table'].lower() == 'panel_embeddings' and t['schema'].lower() in ('testcl', 'test_cl')),
                None
            )
            
            if testcl_table:
                source_schema = testcl_table["schema"]
                source_table = testcl_table["table"]
            else:
                # 첫 번째 발견된 테이블 사용
                source_schema = tables[0]["schema"]
                source_table = tables[0]["table"]
        else:
            # 지정된 테이블 확인
            check_table_sql = f"""
            SELECT EXISTS (
                SELECT 1 
                FROM information_schema.tables 
                WHERE table_schema = :schema 
                AND table_name = :table
            ) as table_exists;
            """
            
            result = await session.execute(text(check_table_sql), {
                "schema": source_schema,
                "table": source_table
            })
            table_exists = result.scalar()
            
            if not table_exists:
                return {
                    "created": False,
                    "error": f"원본 테이블 {source_schema}.{source_table}가 존재하지 않습니다",
                    "table_exists": False
                }
        
        # 2. RawData 스키마 생성 (없으면)
        create_schema_sql = f'CREATE SCHEMA IF NOT EXISTS "{DBN.RAW}";'
        await session.execute(text(create_schema_sql))
        await session.commit()
        
        # 3. 뷰 생성
        EMB_V = fq(DBN.RAW, DBN.T_EMB_V)
        SOURCE_TABLE = fq(source_schema, source_table)
        
        # 원본 테이블의 컬럼 확인
        check_columns_sql = f"""
        SELECT column_name
        FROM information_schema.columns 
        WHERE table_schema = :schema 
        AND table_name = :table
        ORDER BY ordinal_position;
        """
        
        result = await session.execute(text(check_columns_sql), {
            "schema": source_schema,
            "table": source_table
        })
        columns = [row[0] for row in result]
        
        # 필요한 컬럼만 선택 (없으면 NULL)
        column_mapping = {
            'mb_sn': None,
            'demographics': None,
            'combined_text': None,
            'labeled_text': None,
            'chunks': None,
            'chunk_count': None,
            'categories': None,
            'all_labels': None,
            'embedding': None,
            'created_at': None
        }
        
        select_parts = []
        for col_name in column_mapping.keys():
            if col_name in columns:
                select_parts.append(f"{col_name}")
            else:
                select_parts.append(f"NULL as {col_name}")
        
        create_view_sql = f"""
        CREATE OR REPLACE VIEW {EMB_V} AS
        SELECT
            {', '.join(select_parts)}
        FROM {SOURCE_TABLE};
        """
        
        await session.execute(text(create_view_sql))
        await session.commit()
        
        # 4. 뷰 주석 추가
        comment_sql = f"""
        COMMENT ON VIEW {EMB_V} IS 
            'Bridge view from {SOURCE_TABLE} to {DBN.RAW} schema for consistency';
        """
        await session.execute(text(comment_sql))
        await session.commit()
        
        return {
            "created": True,
            "view_name": EMB_V,
            "source_table": SOURCE_TABLE,
            "source_schema": source_schema,
            "source_table_name": source_table,
            "available_columns": columns
        }
        
    except Exception as e:
        await session.rollback()
        import traceback
        logger = logging.getLogger(__name__)
        logger.error(f"뷰 생성 실패: {str(e)}", exc_info=True)
        
        return {
            "created": False,
            "error": str(e),
            "error_type": type(e).__name__
        }


async def test_vector_db_connection(session: AsyncSession) -> Dict[str, Any]:
    """
    벡터 DB 연결 테스트 및 메타데이터 확인
    
    Returns:
        연결 상태 및 메타데이터 딕셔너리
    """
    try:
        # search_path 설정
        await session.execute(text(f'SET search_path TO "{DBN.RAW}", public'))
        
        # 1. 뷰 존재 확인
        EMB_V = fq(DBN.RAW, DBN.T_EMB_V)
        
        check_view_sql = f"""
        SELECT EXISTS (
            SELECT 1 
            FROM information_schema.views 
            WHERE table_schema = '{DBN.RAW}' 
            AND table_name = '{DBN.T_EMB_V}'
        ) as view_exists;
        """
        
        result = await session.execute(text(check_view_sql))
        view_exists = result.scalar()
        # 뷰가 없으면 자동 생성 시도
        if not view_exists:
            create_result = await create_panel_embeddings_view(session)
            if not create_result.get("created", False):
                return {
                    "connected": False,
                    "error": create_result.get("error", "뷰 생성 실패"),
                    "view_exists": False,
                    "auto_create_attempted": True,
                    "create_result": create_result
                }
            # 뷰 생성 후 다시 확인
            result = await session.execute(text(check_view_sql))
            view_exists = result.scalar()
        
        # 2. embedding 컬럼 존재 확인
        check_columns_sql = f"""
        SELECT column_name, data_type 
        FROM information_schema.columns 
        WHERE table_schema = '{DBN.RAW}' 
        AND table_name = '{DBN.T_EMB_V}'
        AND column_name = 'embedding';
        """
        
        result = await session.execute(text(check_columns_sql))
        embedding_col = result.first()
        
        if not embedding_col:
            return {
                "connected": False,
                "error": f"뷰 {EMB_V}에 embedding 컬럼이 없습니다",
                "view_exists": True,
                "embedding_column_exists": False
            }
        
        # 3. 샘플 데이터 확인 (embedding 있는 행 수)
        count_sql = f"""
        SELECT 
            COUNT(*) as total_rows,
            COUNT(embedding) as rows_with_embedding,
            COUNT(DISTINCT mb_sn) as unique_panels
        FROM {EMB_V};
        """
        
        result = await session.execute(text(count_sql))
        row = result.first()
        total_rows = row[0] if row else 0
        rows_with_embedding = row[1] if row else 0
        unique_panels = row[2] if row else 0
        
        # 4. embedding 컬럼 타입 및 차원 확인
        check_embedding_type_sql = f"""
        SELECT 
            data_type,
            udt_name
        FROM information_schema.columns 
        WHERE table_schema = '{DBN.RAW}' 
        AND table_name = '{DBN.T_EMB_V}'
        AND column_name = 'embedding';
        """
        
        result = await session.execute(text(check_embedding_type_sql))
        type_row = result.first()
        embedding_type = type_row[0] if type_row else None
        udt_name = type_row[1] if type_row and len(type_row) > 1 else None
        
        embedding_dim = None
        # pgvector의 vector 타입에서 차원 추출 (올바른 방법)
        if udt_name == 'vector':
            # pgvector vector 타입의 차원 확인 방법
            # 방법 1: 샘플 데이터를 텍스트로 가져와서 차원 추출 (가장 안전)
            try:
                sample_sql = f"""
                SELECT embedding::text
                FROM {EMB_V} 
                WHERE embedding IS NOT NULL 
                LIMIT 1;
                """
                result = await session.execute(text(sample_sql))
                sample_row = result.first()
                if sample_row and sample_row[0]:
                    # '[1,2,3]' 형태에서 차원 추출
                    vec_text = sample_row[0].strip('[]')
                    if vec_text:
                        embedding_dim = len([x for x in vec_text.split(',') if x.strip()])
            except Exception as e:
                await session.rollback()
                
                # 방법 2: pgvector의 vector 차원 함수 사용 (있는 경우)
                try:
                    dim_sql = f"""
                    SELECT array_length(
                        string_to_array(
                            trim(both '[]' from embedding::text),
                            ','
                        ),
                        1
                    ) as dim
                    FROM {EMB_V} 
                    WHERE embedding IS NOT NULL 
                    LIMIT 1;
                    """
                    result = await session.execute(text(dim_sql))
                    dim_row = result.first()
                    if dim_row:
                        embedding_dim = dim_row[0]
                except Exception as e2:
                    await session.rollback()
                    # 차원을 알 수 없으면 None 유지
        
        # 5. pgvector 확장 확인 (트랜잭션이 괜찮은 상태에서만)
        extension_exists = False
        try:
            check_extension_sql = """
            SELECT EXISTS (
                SELECT 1 
                FROM pg_extension 
                WHERE extname = 'vector'
            ) as extension_exists;
            """
            
            result = await session.execute(text(check_extension_sql))
            extension_exists = result.scalar()
        except Exception as e:
            # 확장 확인 실패해도 계속 진행
            try:
                await session.rollback()
            except:
                pass
        
        result_data = {
            "connected": True,
            "view_exists": True,
            "embedding_column_exists": True,
            "extension_exists": extension_exists,
            "total_rows": total_rows,
            "rows_with_embedding": rows_with_embedding,
            "unique_panels": unique_panels,
            "embedding_dimension": embedding_dim,
            "view_name": EMB_V
        }
        
        return result_data
        
    except Exception as e:
        import traceback
        logger = logging.getLogger(__name__)
        logger.error(f"벡터 DB 연결 테스트 실패: {str(e)}", exc_info=True)
        
        # 트랜잭션 롤백 시도
        try:
            await session.rollback()
        except:
            pass
        
        return {
            "connected": False,
            "error": str(e),
            "error_type": type(e).__name__
        }


async def search_panels_by_embedding(
    session: AsyncSession,
    query_embedding: List[float],
    limit: int = 20,
    filters: Optional[Dict[str, Any]] = None
) -> List[Dict[str, Any]]:
    """
    임베딩 벡터로 패널 검색 (cosine similarity)
    
    Args:
        session: 비동기 데이터베이스 세션
        query_embedding: 쿼리 텍스트의 임베딩 벡터
        limit: 반환할 최대 결과 수
        filters: 추가 필터 (성별, 지역 등)
        
    Returns:
        검색 결과 리스트 (유사도 점수 포함)
    """
    try:
        # pgvector 확장 확인 및 활성화 (필요한 경우)
        # Neon DB에서는 확장이 이미 설치되어 있을 가능성이 높음
        try:
            # 먼저 확장 설치 여부 확인
            check_ext_sql = "SELECT EXISTS(SELECT 1 FROM pg_extension WHERE extname = 'vector') as ext_exists;"
            result = await session.execute(text(check_ext_sql))
            ext_exists = result.scalar()
            
            if not ext_exists:
                # 확장이 없으면 활성화 시도
                try:
                    await session.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
                    await session.commit()
                except Exception as create_error:
                    pass  # 권한 문제일 수 있음, 계속 진행
            
            # vector 타입 존재 여부 확인
            check_vector_type_sql = """
            SELECT EXISTS(
                SELECT 1 FROM pg_type WHERE typname = 'vector'
            ) as vector_type_exists;
            """
            result = await session.execute(text(check_vector_type_sql))
            vector_type_exists = result.scalar()
            
            if not vector_type_exists:
                raise RuntimeError("pgvector 확장이 활성화되지 않았거나 vector 타입을 찾을 수 없습니다.")
        except Exception as ext_error:
            # 확장 확인 실패해도 계속 진행 (이미 설치되어 있을 수 있음)
            # 하지만 vector 타입이 없으면 나중에 에러 발생
        
        # search_path 설정 (testcl 스키마도 포함, public 스키마는 선택적)
        # Neon DB에서는 public 스키마가 없을 수 있으므로, vector 타입이 있는 곳을 찾아야 함
        try:
            # vector 타입이 있는 스키마 찾기
            find_vector_schema_sql = """
            SELECT nspname 
            FROM pg_type t
            JOIN pg_namespace n ON t.typnamespace = n.oid
            WHERE t.typname = 'vector'
            LIMIT 1;
            """
            result = await session.execute(text(find_vector_schema_sql))
            vector_schema = result.scalar()
            
            if vector_schema:
                # vector 타입이 있는 스키마를 search_path에 포함
                search_path = f'"{DBN.RAW}", "testcl", "{vector_schema}"'
            else:
                search_path = f'"{DBN.RAW}", "testcl"'
        except Exception as schema_error:
            search_path = f'"{DBN.RAW}", "testcl"'
        
        await session.execute(text(f'SET search_path TO {search_path}'))
        
        # 뷰 존재 확인
        EMB_V = fq(DBN.RAW, DBN.T_EMB_V)
        check_view_sql = f"""
        SELECT EXISTS (
            SELECT 1 
            FROM information_schema.views 
            WHERE table_schema = '{DBN.RAW}' 
            AND table_name = '{DBN.T_EMB_V}'
        ) as view_exists;
        """
        
        result = await session.execute(text(check_view_sql))
        view_exists = result.scalar()
        
        # 뷰가 없으면 자동 생성 시도
        if not view_exists:
            create_result = await create_panel_embeddings_view(session)
            if not create_result.get("created", False):
                error_msg = f"뷰 {EMB_V}가 존재하지 않으며 생성에도 실패했습니다: {create_result.get('error', '알 수 없는 오류')}"
                raise RuntimeError(error_msg)
        
        # 임베딩 벡터를 PostgreSQL 배열 형식으로 변환 (소수점 6자리)
        embedding_str = '[' + ','.join(f"{x:.6f}" for x in query_embedding) + ']'
        
        # 기본 벡터 검색 쿼리 (cosine similarity)
        # pgvector의 <#> 연산자는 cosine distance를 반환 (코사인 거리)
        # similarity = 1 - distance (코사인 유사도)
        # vector 타입은 스키마 미지정 (pgvector 타입은 전역 등록됨)
        # 유사도 0.9 이상만 필터링
        base_sql = f"""
        SELECT 
            mb_sn,
            demographics,
            combined_text,
            labeled_text,
            chunks,
            chunk_count,
            categories,
            all_labels,
            embedding,
            1 - (embedding <#> CAST(:query_embedding AS vector)) AS similarity,
            (embedding <#> CAST(:query_embedding AS vector)) AS distance
        FROM {EMB_V}
        WHERE embedding IS NOT NULL
        AND 1 - (embedding <#> CAST(:query_embedding AS vector)) >= 0.9
        """
        
        params = {
            "query_embedding": embedding_str
        }
        
        # 추가 필터 적용 (RawData 테이블과 JOIN 필요)
        # 벡터 검색 결과에 대해 필터를 적용하려면 RawData 테이블과 JOIN 필요
        # 하지만 성능을 위해 벡터 검색 후 Python에서 필터링하는 것이 더 효율적일 수 있음
        # 일단 demographics JSONB에서 가능한 필터만 적용
        
        # 성별 필터 (demographics JSONB에서)
        if filters and (gender := filters.get("gender")):
            if isinstance(gender, list):
                gender_conditions = []
                for idx, g in enumerate(gender):
                    # 성별 정규화
                    g_lower = str(g).lower()
                    gender_map = {"m": "남성", "f": "여성", "male": "남성", "female": "여성", "남": "남성", "여": "여성"}
                    g_normalized = gender_map.get(g_lower, g)
                    
                    gender_conditions.append(
                        f"(demographics->>'gender' = :g{idx}_1 OR demographics->>'gender' = :g{idx}_2 OR "
                        f"LOWER(COALESCE(demographics->>'gender', '')) = :g{idx}_3)"
                    )
                    params[f"g{idx}_1"] = g_normalized
                    params[f"g{idx}_2"] = g_lower
                    params[f"g{idx}_3"] = g_lower
                if gender_conditions:
                    base_sql += " AND (" + " OR ".join(gender_conditions) + ")"
            else:
                g_lower = str(gender).lower()
                gender_map = {"m": "남성", "f": "여성", "male": "남성", "female": "여성", "남": "남성", "여": "여성"}
                g_normalized = gender_map.get(g_lower, gender)
                base_sql += " AND (demographics->>'gender' = :g1 OR demographics->>'gender' = :g2 OR LOWER(COALESCE(demographics->>'gender', '')) = :g3)"
                params["g1"] = g_normalized
                params["g2"] = g_lower
                params["g3"] = g_lower
        
        # 지역 필터 (demographics JSONB에서)
        if filters and (region := filters.get("region")):
            if isinstance(region, list):
                region_conditions = []
                for idx, r in enumerate(region):
                    region_conditions.append(f"LOWER(COALESCE(demographics->>'region', demographics->>'location', '')) LIKE :r{idx}")
                    params[f"r{idx}"] = f"%{str(r).lower()}%"
                if region_conditions:
                    base_sql += " AND (" + " OR ".join(region_conditions) + ")"
            else:
                base_sql += " AND LOWER(COALESCE(demographics->>'region', demographics->>'location', '')) LIKE :r"
                params["r"] = f"%{str(region).lower()}%"
        
        # 나이 필터는 demographics에서 나이 정보가 있을 경우에만 적용 가능
        # 실제로는 벡터 검색 후 RawData와 JOIN해서 나이 계산 필요
        if filters and (age_min := filters.get("age_min")):
            base_sql += " AND (demographics->>'age')::int >= :age_min"
            params["age_min"] = int(age_min)
        if filters and (age_max := filters.get("age_max")):
            base_sql += " AND (demographics->>'age')::int <= :age_max"
            params["age_max"] = int(age_max)
        
        # 거리 오름차순으로 정렬하고 LIMIT (코사인 거리 기준, 작을수록 유사)
        base_sql += " ORDER BY distance ASC LIMIT :limit"
        params["limit"] = limit
        
        result = await session.execute(text(base_sql), params)
        rows = [dict(row) for row in result.mappings()]
        
        return rows
        
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error(f"벡터 검색 실패: {str(e)}", exc_info=True)
        raise

