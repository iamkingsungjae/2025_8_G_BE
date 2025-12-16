"""
HDBSCAN 알고리즘 구현
"""

from typing import Any, Dict, Optional
import numpy as np
try:
    from hdbscan import HDBSCAN as SklearnHDBSCAN
    HDBSCAN_AVAILABLE = True
except ImportError:
    HDBSCAN_AVAILABLE = False
    SklearnHDBSCAN = None

from .base import BaseClusteringAlgorithm


class HDBSCANAlgorithm(BaseClusteringAlgorithm):
    """HDBSCAN 클러스터링 알고리즘"""
    
    def __init__(self, 
                 min_cluster_size: int = 5,
                 min_samples: Optional[int] = None,
                 metric: str = 'euclidean'):
        """
        Parameters:
        -----------
        min_cluster_size : int
            최소 클러스터 크기
        min_samples : int, optional
            최소 샘플 수 (None이면 min_cluster_size와 동일)
        metric : str
            거리 메트릭
        """
        if not HDBSCAN_AVAILABLE:
            raise ImportError("hdbscan 패키지가 설치되지 않았습니다. pip install hdbscan")
        
        self.min_cluster_size = min_cluster_size
        self.min_samples = min_samples or min_cluster_size
        self.metric = metric
        self.model: Optional[SklearnHDBSCAN] = None
    
    def fit(self, X: np.ndarray, **kwargs) -> 'HDBSCANAlgorithm':
        """모델 학습"""
        min_cluster_size = kwargs.get('min_cluster_size', self.min_cluster_size)
        min_samples = kwargs.get('min_samples', self.min_samples)
        metric = kwargs.get('metric', self.metric)
        
        self.model = SklearnHDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            metric=metric
        )
        self.model.fit(X)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """클러스터 예측 (HDBSCAN은 fit_predict만 지원)"""
        if self.model is None:
            raise ValueError("모델이 학습되지 않았습니다. fit()을 먼저 호출하세요.")
        # HDBSCAN은 별도 predict가 없으므로 approximate_predict 사용
        labels, strengths = self.model.approximate_predict(X)
        return labels
    
    def fit_predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """학습 및 예측"""
        self.fit(X, **kwargs)
        return self.model.labels_
    
    def get_algorithm_info(self) -> Dict[str, Any]:
        """알고리즘 정보 반환"""
        return {
            'type': 'HDBSCAN',
            'min_cluster_size': self.min_cluster_size,
            'min_samples': self.min_samples,
            'metric': self.metric,
            'is_fitted': self.model is not None,
            'n_clusters': len(set(self.model.labels_)) - (1 if -1 in self.model.labels_ else 0) if self.model else None
        }




