"""
벡터 프로세서 구현
벡터 데이터 처리 및 변환
"""

from typing import Any, Dict, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from .base import BaseProcessor


class VectorProcessor(BaseProcessor):
    """벡터 데이터 프로세서"""
    
    def __init__(self, 
                 scaler_type: str = 'standard',
                 handle_missing: str = 'mean'):
        """
        Parameters:
        -----------
        scaler_type : str
            스케일러 타입 ('standard', 'minmax', 'none')
        handle_missing : str
            결측치 처리 방법 ('mean', 'median', 'zero', 'drop')
        """
        self.scaler_type = scaler_type
        self.handle_missing = handle_missing
        self.scaler = None
        self.feature_names: Optional[list] = None
    
    def process(
        self, 
        data: pd.DataFrame, 
        features: Optional[list] = None,
        **kwargs
    ) -> np.ndarray:
        """
        벡터 데이터 처리
        
        Parameters:
        -----------
        data : pd.DataFrame
            처리할 데이터
        features : list, optional
            사용할 피쳐 목록 (None이면 모든 숫자형 컬럼 사용)
        **kwargs
            추가 파라미터
        
        Returns:
        --------
        np.ndarray
            처리된 벡터 데이터
        """
        # 피쳐 선택
        if features is None:
            # 숫자형 컬럼만 선택
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            features = numeric_cols
        else:
            # 존재하는 피쳐만 선택
            features = [f for f in features if f in data.columns]
        
        if not features:
            raise ValueError("사용 가능한 피쳐가 없습니다.")
        
        self.feature_names = features
        X = data[features].copy()
        
        # 결측치 처리
        if X.isnull().sum().sum() > 0:
            if self.handle_missing == 'mean':
                X = X.fillna(X.mean())
            elif self.handle_missing == 'median':
                X = X.fillna(X.median())
            elif self.handle_missing == 'zero':
                X = X.fillna(0)
            elif self.handle_missing == 'drop':
                X = X.dropna()
        
        # 스케일링
        if self.scaler_type == 'standard':
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        elif self.scaler_type == 'minmax':
            self.scaler = MinMaxScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:  # 'none'
            X_scaled = X.values
        
        return X_scaled
    
    def transform(self, data: pd.DataFrame) -> np.ndarray:
        """
        학습된 스케일러로 변환 (fit 없이)
        
        Parameters:
        -----------
        data : pd.DataFrame
            변환할 데이터
        
        Returns:
        --------
        np.ndarray
            변환된 데이터
        """
        if self.scaler is None:
            raise ValueError("스케일러가 학습되지 않았습니다. process()를 먼저 호출하세요.")
        
        if self.feature_names is None:
            raise ValueError("피쳐 목록이 설정되지 않았습니다.")
        
        X = data[self.feature_names].copy()
        
        # 결측치 처리 (학습 시와 동일)
        if X.isnull().sum().sum() > 0:
            if self.handle_missing == 'mean':
                X = X.fillna(X.mean())
            elif self.handle_missing == 'median':
                X = X.fillna(X.median())
            elif self.handle_missing == 'zero':
                X = X.fillna(0)
            elif self.handle_missing == 'drop':
                X = X.dropna()
        
        return self.scaler.transform(X)
    
    def get_processor_info(self) -> Dict[str, Any]:
        """프로세서 정보 반환"""
        return {
            'type': 'VectorProcessor',
            'scaler_type': self.scaler_type,
            'handle_missing': self.handle_missing,
            'feature_names': self.feature_names,
            'is_fitted': self.scaler is not None
        }




