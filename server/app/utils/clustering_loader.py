"""클러스터링 데이터 NeonDB 로더"""
import os
import sys
import json
import logging
import asyncio
from typing import Dict, Any, Optional, List
from pathlib import Path
import pandas as pd
import numpy as np
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

# Windows 이벤트 루프 정책 설정
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def _is_valid_uuid(session_id: str) -> bool:
    """UUID 형식 검증 헬퍼 함수"""
    import re
    uuid_pattern = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.IGNORECASE)
    return bool(uuid_pattern.match(session_id))


def _get_db_session():
    """NeonDB 세션 생성 (check_clustering_db.py와 동일한 방식)"""
    load_dotenv(override=True)
    uri = os.getenv("ASYNC_DATABASE_URI")
    if not uri:
        logger.error("[Clustering Loader] ASYNC_DATABASE_URI 환경변수가 설정되지 않았습니다.")
        return None, None
    
    # postgresql://를 postgresql+psycopg://로 변환
    if uri.startswith("postgresql://"):
        uri = uri.replace("postgresql://", "postgresql+psycopg://", 1)
    elif "postgresql+asyncpg" in uri:
        uri = uri.replace("postgresql+asyncpg", "postgresql+psycopg", 1)
    
    engine = create_async_engine(uri, echo=False, pool_pre_ping=True)
    SessionLocal = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)
    return engine, SessionLocal


async def load_clustering_session_from_db(session_id: str) -> Optional[Dict[str, Any]]:
    """
    NeonDB에서 클러스터링 세션 정보 로드
    
    Returns:
        세션 정보 딕셔너리 또는 None
    """
    logger.info(f"[Clustering Loader] 세션 로드 시작: session_id={session_id}")
    
    # UUID 형식 검증
    if not _is_valid_uuid(session_id):
        logger.warning(f"[Clustering Loader] UUID 형식이 아닌 session_id: {session_id}, None 반환")
        return None
    
    engine, SessionLocal = _get_db_session()
    if not engine or not SessionLocal:
        logger.error("[Clustering Loader] DB 세션 생성 실패")
        return None
    
    try:
        async with SessionLocal() as session:
            # merged 스키마에서 세션 정보 조회
            result = await session.execute(
                text("""
                    SELECT 
                        session_id, created_at, updated_at,
                        n_samples, n_clusters, algorithm, optimal_k, strategy,
                        silhouette_score, davies_bouldin_score, calinski_harabasz_score,
                        request_params, feature_types, algorithm_info,
                        filter_info, processor_info, is_precomputed, precomputed_name
                    FROM merged.clustering_sessions
                    WHERE session_id = :session_id
                """),
                {"session_id": session_id}
            )
            row = result.fetchone()
            
            if not row:
                logger.warning(f"[Clustering Loader] 세션을 찾을 수 없음: session_id={session_id}")
                return None
            
            logger.info(f"[Clustering Loader] 세션 정보 로드 완료: session_id={session_id}, n_samples={row.n_samples}, n_clusters={row.n_clusters}")
            
            # 딕셔너리로 변환
            session_data = {
                'session_id': str(row.session_id),
                'created_at': row.created_at.isoformat() if row.created_at else None,
                'updated_at': row.updated_at.isoformat() if row.updated_at else None,
                'n_samples': row.n_samples,
                'n_clusters': row.n_clusters,
                'algorithm': row.algorithm,
                'optimal_k': row.optimal_k,
                'strategy': row.strategy,
                'silhouette_score': float(row.silhouette_score) if row.silhouette_score is not None else None,
                'davies_bouldin_score': float(row.davies_bouldin_score) if row.davies_bouldin_score is not None else None,
                'calinski_harabasz_score': float(row.calinski_harabasz_score) if row.calinski_harabasz_score is not None else None,
                'request_params': row.request_params if isinstance(row.request_params, dict) else json.loads(row.request_params) if row.request_params else {},
                'feature_types': row.feature_types if isinstance(row.feature_types, dict) else json.loads(row.feature_types) if row.feature_types else {},
                'algorithm_info': row.algorithm_info if isinstance(row.algorithm_info, dict) else json.loads(row.algorithm_info) if row.algorithm_info else {},
                'filter_info': row.filter_info if isinstance(row.filter_info, dict) else json.loads(row.filter_info) if row.filter_info else {},
                'processor_info': row.processor_info if isinstance(row.processor_info, dict) else json.loads(row.processor_info) if row.processor_info else {},
                'is_precomputed': row.is_precomputed,
                'precomputed_name': row.precomputed_name,
            }
            
            return session_data
            
    except Exception as e:
        logger.error(f"[Clustering Loader] 세션 로드 실패: session_id={session_id}, 오류: {str(e)}", exc_info=True)
        return None
    finally:
        if engine:
            await engine.dispose()


async def load_panel_cluster_mappings_from_db(session_id: str) -> Optional[pd.DataFrame]:
    """
    NeonDB에서 패널-클러스터 매핑 로드
    
    Returns:
        DataFrame (mb_sn, cluster_id) 또는 None
    """
    logger.info(f"[Clustering Loader] 매핑 로드 시작: session_id={session_id}")
    
    # UUID 형식 검증
    import re
    uuid_pattern = re.compile(r'^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$', re.IGNORECASE)
    if not uuid_pattern.match(session_id):
        logger.warning(f"[Clustering Loader] UUID 형식이 아닌 session_id: {session_id}, None 반환")
        return None
    
    engine, SessionLocal = _get_db_session()
    if not engine or not SessionLocal:
        logger.error("[Clustering Loader] DB 세션 생성 실패")
        return None
    
    try:
        async with SessionLocal() as session:
            # merged 스키마에서 매핑 조회
            result = await session.execute(
                text("""
                    SELECT mb_sn, cluster_id
                    FROM merged.panel_cluster_mappings
                    WHERE session_id = :session_id
                    ORDER BY mb_sn
                """),
                {"session_id": session_id}
            )
            rows = result.fetchall()
            
            if not rows:
                logger.warning(f"[Clustering Loader] 매핑 데이터 없음: session_id={session_id}")
                return None
            
            logger.info(f"[Clustering Loader] 매핑 로드 완료: {len(rows)}개 매핑")
            
            # DataFrame으로 변환
            df = pd.DataFrame([{'mb_sn': str(row.mb_sn), 'cluster': int(row.cluster_id)} for row in rows])
            
            return df
            
    except Exception as e:
        logger.error(f"[Clustering Loader] 매핑 로드 실패: session_id={session_id}, 오류: {str(e)}", exc_info=True)
        return None
    finally:
        if engine:
            await engine.dispose()


async def load_umap_coordinates_from_db(session_id: str) -> Optional[pd.DataFrame]:
    """
    NeonDB에서 UMAP 좌표 로드
    
    Returns:
        DataFrame (mb_sn, umap_x, umap_y) 또는 None
    """
    logger.info(f"[Clustering Loader] UMAP 좌표 로드 시작: session_id={session_id}")
    
    # UUID 형식 검증
    if not _is_valid_uuid(session_id):
        logger.warning(f"[Clustering Loader] UUID 형식이 아닌 session_id: {session_id}, None 반환")
        return None
    
    engine, SessionLocal = _get_db_session()
    if not engine or not SessionLocal:
        logger.error("[Clustering Loader] DB 세션 생성 실패")
        return None
    
    try:
        async with SessionLocal() as session:
            # merged 스키마에서 UMAP 좌표 조회
            result = await session.execute(
                text("""
                    SELECT mb_sn, umap_x, umap_y
                    FROM merged.umap_coordinates
                    WHERE session_id = :session_id
                    ORDER BY mb_sn
                """),
                {"session_id": session_id}
            )
            rows = result.fetchall()
            
            if not rows:
                logger.warning(f"[Clustering Loader] UMAP 좌표 데이터 없음: session_id={session_id}")
                return None
            
            logger.info(f"[Clustering Loader] UMAP 좌표 로드 완료: {len(rows)}개 좌표")
            
            # DataFrame으로 변환
            df = pd.DataFrame([
                {
                    'mb_sn': str(row.mb_sn),
                    'umap_x': float(row.umap_x),
                    'umap_y': float(row.umap_y)
                }
                for row in rows
            ])
            
            return df
            
    except Exception as e:
        logger.error(f"[Clustering Loader] UMAP 좌표 로드 실패: session_id={session_id}, 오류: {str(e)}", exc_info=True)
        return None
    finally:
        if engine:
            await engine.dispose()


async def load_cluster_profiles_from_db(session_id: str) -> Optional[List[Dict[str, Any]]]:
    """
    NeonDB에서 클러스터 프로필 로드
    
    Returns:
        프로필 리스트 또는 None
    """
    logger.info(f"[Clustering Loader] 클러스터 프로필 로드 시작: session_id={session_id}")
    
    # UUID 형식 검증
    if not _is_valid_uuid(session_id):
        logger.warning(f"[Clustering Loader] UUID 형식이 아닌 session_id: {session_id}, None 반환")
        return None
    
    engine, SessionLocal = _get_db_session()
    if not engine or not SessionLocal:
        logger.error("[Clustering Loader] DB 세션 생성 실패")
        return None
    
    try:
        async with SessionLocal() as session:
            # merged 스키마에서 프로필 조회
            result = await session.execute(
                text("""
                    SELECT 
                        cluster_id, size, percentage, name, tags,
                        distinctive_features, insights, insights_by_category,
                        segments, features
                    FROM merged.cluster_profiles
                    WHERE session_id = :session_id
                    ORDER BY cluster_id
                """),
                {"session_id": session_id}
            )
            rows = result.fetchall()
            
            if not rows:
                logger.warning(f"[Clustering Loader] 프로필 데이터 없음: session_id={session_id}")
                return []
            
            logger.info(f"[Clustering Loader] 프로필 로드 완료: {len(rows)}개 프로필")
            
            # 딕셔너리 리스트로 변환
            profiles = []
            for row in rows:
                segments = row.segments if isinstance(row.segments, dict) else json.loads(row.segments) if row.segments else {}
                
                profile = {
                    'cluster': int(row.cluster_id),
                    'size': int(row.size),
                    'percentage': float(row.percentage),
                    'name': row.name,
                    'tags': row.tags if isinstance(row.tags, list) else list(row.tags) if row.tags else [],
                    'distinctive_features': row.distinctive_features if isinstance(row.distinctive_features, list) else (json.loads(row.distinctive_features) if isinstance(row.distinctive_features, str) else (list(row.distinctive_features) if row.distinctive_features else [])),
                    'insights': row.insights if isinstance(row.insights, list) else list(row.insights) if row.insights else [],
                    'insights_by_category': row.insights_by_category if isinstance(row.insights_by_category, dict) else json.loads(row.insights_by_category) if row.insights_by_category else {},
                    'segments': segments,
                    'features': row.features if isinstance(row.features, dict) else json.loads(row.features) if row.features else {},
                }
                
                # segments에서 새로운 필드들을 최상위 레벨로 추출
                if isinstance(segments, dict):
                    profile['name_main'] = segments.get('name_main', row.name or '')
                    profile['name_sub'] = segments.get('name_sub', '')
                    profile['tags_hierarchical'] = segments.get('tags_hierarchical', {})
                    profile['insights_storytelling'] = segments.get('insights_storytelling', {})
                else:
                    profile['name_main'] = row.name or ''
                    profile['name_sub'] = ''
                    profile['tags_hierarchical'] = {}
                    profile['insights_storytelling'] = {}
                
                profiles.append(profile)
            
            return profiles
            
    except Exception as e:
        logger.error(f"[Clustering Loader] 프로필 로드 실패: session_id={session_id}, 오류: {str(e)}", exc_info=True)
        return None
    finally:
        if engine:
            await engine.dispose()


async def get_precomputed_session_id(precomputed_name: str = "hdbscan_default") -> Optional[str]:
    """
    Precomputed 세션 ID 조회
    
    Returns:
        session_id 또는 None
    """
    logger.info(f"[Clustering Loader] Precomputed 세션 ID 조회: name={precomputed_name}")
    
    engine, SessionLocal = _get_db_session()
    if not engine or not SessionLocal:
        logger.error("[Clustering Loader] DB 세션 생성 실패")
        return None
    
    try:
        async with SessionLocal() as session:
            # merged 스키마에서 Precomputed 세션 조회
            result = await session.execute(
                text("""
                    SELECT session_id
                    FROM merged.clustering_sessions
                    WHERE is_precomputed = TRUE AND precomputed_name = :precomputed_name
                    ORDER BY created_at DESC
                    LIMIT 1
                """),
                {"precomputed_name": precomputed_name}
            )
            row = result.fetchone()
            
            if not row:
                logger.warning(f"[Clustering Loader] Precomputed 세션을 찾을 수 없음: name={precomputed_name}")
                return None
            
            session_id = str(row.session_id)
            logger.info(f"[Clustering Loader] Precomputed 세션 ID 찾음: {session_id}")
            return session_id
            
    except Exception as e:
        logger.error(f"[Clustering Loader] Precomputed 세션 ID 조회 실패: name={precomputed_name}, 오류: {str(e)}", exc_info=True)
        return None
    finally:
        if engine:
            await engine.dispose()


async def load_full_clustering_data_from_db(session_id: str) -> Optional[Dict[str, Any]]:
    """
    NeonDB에서 전체 클러스터링 데이터 로드 (세션, 매핑, 좌표 통합)
    
    Returns:
        artifacts 딕셔너리 (data, labels, meta 포함) 또는 None
    """
    logger.info(f"[Clustering Loader] 전체 클러스터링 데이터 로드 시작: session_id={session_id}")
    
    # 1. 세션 정보 로드
    session_data = await load_clustering_session_from_db(session_id)
    if not session_data:
        logger.error(f"[Clustering Loader] 세션 정보 로드 실패: session_id={session_id}")
        return None
    
    # 2. 매핑 로드
    mappings_df = await load_panel_cluster_mappings_from_db(session_id)
    if mappings_df is None:
        logger.warning(f"[Clustering Loader] 매핑 데이터 없음, 빈 DataFrame 사용: session_id={session_id}")
        mappings_df = pd.DataFrame(columns=['mb_sn', 'cluster'])
    
    # 3. UMAP 좌표 로드
    umap_df = await load_umap_coordinates_from_db(session_id)
    if umap_df is None:
        logger.warning(f"[Clustering Loader] UMAP 좌표 데이터 없음, 빈 DataFrame 사용: session_id={session_id}")
        umap_df = pd.DataFrame(columns=['mb_sn', 'umap_x', 'umap_y'])
    
    # 4. 데이터 병합 (mb_sn 기준)
    logger.info(f"[Clustering Loader] 데이터 병합 시작: 매핑={len(mappings_df)}행, 좌표={len(umap_df)}행")
    
    # mb_sn 기준으로 병합
    if not mappings_df.empty and not umap_df.empty:
        data_df = mappings_df.merge(umap_df, on='mb_sn', how='outer')
    elif not mappings_df.empty:
        data_df = mappings_df.copy()
        data_df['umap_x'] = None
        data_df['umap_y'] = None
    elif not umap_df.empty:
        data_df = umap_df.copy()
        data_df['cluster'] = -1
    else:
        logger.error(f"[Clustering Loader] 병합할 데이터가 없음: session_id={session_id}")
        return None
    
    logger.info(f"[Clustering Loader] 데이터 병합 완료: {len(data_df)}행")
    
    # 5. labels 배열 생성 (mb_sn 순서 유지)
    labels = data_df['cluster'].values if 'cluster' in data_df.columns else np.array([])
    
    # 6. 메타데이터 구성
    meta = {
        'session_id': session_id,
        'result_meta': {
            'n_samples': session_data['n_samples'],
            'n_clusters': session_data['n_clusters'],
            'algorithm': session_data['algorithm'],
            'optimal_k': session_data['optimal_k'],
            'strategy': session_data['strategy'],
            'silhouette_score': session_data['silhouette_score'],
            'davies_bouldin_score': session_data['davies_bouldin_score'],
            'calinski_harabasz_score': session_data['calinski_harabasz_score'],
            'algorithm_info': session_data['algorithm_info'],
            'filter_info': session_data['filter_info'],
            'processor_info': session_data['processor_info'],
            'feature_types': session_data['feature_types'],
            'request_params': session_data['request_params'],
        }
    }
    
    # 7. artifacts 딕셔너리 구성
    artifacts = {
        'data': data_df,
        'labels': labels,
        'meta': meta,
    }
    
    logger.info(f"[Clustering Loader] 전체 클러스터링 데이터 로드 완료: session_id={session_id}, 데이터={len(data_df)}행, 레이블={len(labels)}개")
    
    return artifacts

