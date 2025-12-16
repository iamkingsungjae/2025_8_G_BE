"""
클러스터링 알고리즘 베이스 클래스
다양한 알고리즘을 통일된 인터페이스로 사용
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import numpy as np


class BaseClusteringAlgorithm(ABC):
    """클러스터링 알고리즘 베이스 클래스"""
    
    @abstractmethod
    def fit(self, X: np.ndarray, **kwargs) -> 'BaseClusteringAlgorithm':
        """
        클러스터링 모델 학습
        
        Parameters:
        -----------
        X : np.ndarray
            학습 데이터
        **kwargs
            알고리즘별 파라미터
        
        Returns:
        --------
        self
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        클러스터 예측
        
        Parameters:
        -----------
        X : np.ndarray
            예측할 데이터
        
        Returns:
        --------
        np.ndarray
            클러스터 레이블
        """
        pass
    
    @abstractmethod
    def fit_predict(self, X: np.ndarray, **kwargs) -> np.ndarray:
        """
        학습 및 예측을 한 번에 수행
        
        Parameters:
        -----------
        X : np.ndarray
            학습/예측 데이터
        **kwargs
            알고리즘별 파라미터
        
        Returns:
        --------
        np.ndarray
            클러스터 레이블
        """
        pass
    
    @abstractmethod
    def get_algorithm_info(self) -> Dict[str, Any]:
        """
        알고리즘 정보 반환
        
        Returns:
        --------
        dict
            알고리즘 메타데이터
        """
        pass




