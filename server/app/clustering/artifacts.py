"""아티팩트 저장/로드"""
import uuid
import json
import joblib
import logging
import asyncio
from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)

BASE = Path("runs")
BASE.mkdir(exist_ok=True)


def new_session_dir() -> Path:
    """
    새로운 세션 디렉터리 생성
    
    Returns:
    --------
    Path
        세션 디렉터리 경로
    """
    d = BASE / str(uuid.uuid4())
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_artifacts(
    session_dir: Path, 
    df: pd.DataFrame, 
    labels: Optional[np.ndarray], 
    meta: Dict[str, Any]
) -> None:
    """
    아티팩트 저장
    
    Parameters:
    -----------
    session_dir : Path
        세션 디렉터리
    df : pd.DataFrame
        데이터프레임
    labels : np.ndarray, optional
        클러스터 레이블
    meta : dict
        메타데이터
    """
    # 1. 데이터 저장
    if df is not None:
        data_path = session_dir / "data.csv"
        df.to_csv(data_path, index=False, encoding='utf-8-sig')
    
    # 2. 레이블 저장
    if labels is not None:
        labels_path = session_dir / "labels.npy"
        np.save(labels_path, labels)
        
        # CSV로도 저장 (읽기 쉽게)
        labels_df = pd.DataFrame({
            'index': range(len(labels)),
            'cluster': labels
        })
        labels_csv_path = session_dir / "labels.csv"
        labels_df.to_csv(labels_csv_path, index=False)
    
    # 3. 메타데이터 저장
    meta_path = session_dir / "meta.json"
    # numpy 타입을 JSON 직렬화 가능하게 변환
    meta_serializable = _make_json_serializable(meta)
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta_serializable, f, indent=2, ensure_ascii=False)
    
    # 4. 모델 저장 (있는 경우)
    if 'model' in meta:
        model_path = session_dir / "model.joblib"
        joblib.dump(meta['model'], model_path)


def load_artifacts(session_id: str) -> Optional[Dict[str, Any]]:
    """
    아티팩트 로드 (NeonDB 우선, 파일 시스템 fallback)
    
    Parameters:
    -----------
    session_id : str
        세션 ID (UUID)
    
    Returns:
    --------
    dict, optional
        아티팩트 딕셔너리 (None이면 찾을 수 없음)
    """
    logger.info(f"[Artifacts] 아티팩트 로드 시작: session_id={session_id}")
    
    # 1. NeonDB에서 로드 시도
    try:
        from app.utils.clustering_loader import load_full_clustering_data_from_db
        
        logger.debug(f"[Artifacts] NeonDB에서 로드 시도: session_id={session_id}")
        
        # 비동기 함수를 동기적으로 실행
        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        
        artifacts = loop.run_until_complete(load_full_clustering_data_from_db(session_id))
        
        if artifacts:
            logger.info(f"[Artifacts] NeonDB에서 로드 성공: session_id={session_id}")
            return artifacts
        else:
            logger.warning(
                f"[Artifacts] ⚠️ NeonDB에서 세션을 찾을 수 없음, 파일 시스템 fallback 시도: session_id={session_id}\n"
                f"  → 동적 클러스터링 세션은 파일 시스템(runs/)에만 저장됩니다.\n"
                f"  → Precomputed 세션은 NeonDB에 저장되어야 합니다."
            )
    except Exception as e:
        logger.warning(
            f"[Artifacts] ⚠️ NeonDB 로드 실패, 파일 시스템 fallback 시도: session_id={session_id}, 오류: {str(e)}\n"
            f"  → 동적 클러스터링 세션은 파일 시스템(runs/)에만 저장됩니다.\n"
            f"  → Precomputed 세션은 NeonDB에 저장되어야 합니다."
        )
    
    # 2. 파일 시스템에서 로드 (fallback)
    # ⚠️ 주의: 동적 클러스터링 세션은 파일 시스템에만 저장됩니다.
    # Precomputed 세션은 반드시 NeonDB에 저장되어야 하며, 파일 시스템 fallback은 사용되지 않아야 합니다.
    logger.info(f"[Artifacts] 파일 시스템에서 로드 시도: session_id={session_id} (runs/{session_id})")
    session_dir = BASE / session_id
    
    if not session_dir.exists():
        logger.warning(f"[Artifacts] 세션 디렉터리 없음: {session_dir}")
        return None

    artifacts = {}
    
    # 1. 데이터 로드
    data_path = session_dir / "data.csv"
    if data_path.exists():
        logger.debug(f"[Artifacts] 데이터 파일 로드: {data_path}")
        artifacts['data'] = pd.read_csv(data_path)
    else:
        logger.warning(f"[Artifacts] 데이터 파일 없음: {data_path}")
    
    # 2. 레이블 로드
    labels_path = session_dir / "labels.npy"
    if labels_path.exists():
        logger.debug(f"[Artifacts] 레이블 파일 로드 (NPY): {labels_path}")
        artifacts['labels'] = np.load(labels_path)
    else:
        # CSV에서 로드 시도
        labels_csv_path = session_dir / "labels.csv"
        if labels_csv_path.exists():
            logger.debug(f"[Artifacts] 레이블 파일 로드 (CSV): {labels_csv_path}")
            labels_df = pd.read_csv(labels_csv_path)
            artifacts['labels'] = labels_df['cluster'].values
        else:
            logger.warning(f"[Artifacts] 레이블 파일 없음: {labels_path}, {labels_csv_path}")
    
    # 3. 메타데이터 로드
    meta_path = session_dir / "meta.json"
    if meta_path.exists():
        logger.debug(f"[Artifacts] 메타데이터 파일 로드: {meta_path}")
        with open(meta_path, 'r', encoding='utf-8') as f:
            artifacts['meta'] = json.load(f)
    else:
        logger.warning(f"[Artifacts] 메타데이터 파일 없음: {meta_path}")
    
    # 4. 모델 로드 (있는 경우)
    model_path = session_dir / "model.joblib"
    if model_path.exists():
        logger.debug(f"[Artifacts] 모델 파일 로드: {model_path}")
        artifacts['model'] = joblib.load(model_path)
    
    if artifacts:
        logger.info(f"[Artifacts] 파일 시스템에서 로드 성공: session_id={session_id}, 키: {list(artifacts.keys())}")
        return artifacts
    else:
        logger.warning(f"[Artifacts] 아티팩트를 찾을 수 없음: session_id={session_id}")
        return None


def _make_json_serializable(obj: Any) -> Any:
    """JSON 직렬화 가능하게 변환"""
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    elif isinstance(obj, pd.DataFrame):
        return obj.to_dict('records')
    elif isinstance(obj, pd.Series):
        return obj.to_dict()
    else:
        return obj
