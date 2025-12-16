"""
프로세서 베이스 클래스
데이터 전처리 및 변환을 위한 인터페이스
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import pandas as pd
import numpy as np


class BaseProcessor(ABC):
    """프로세서 베이스 클래스"""
    
    @abstractmethod
    def process(self, data: pd.DataFrame, **kwargs) -> np.ndarray:
        """
        데이터 처리
        
        Parameters:
        -----------
        data : pd.DataFrame
            처리할 데이터
        **kwargs
            처리 파라미터
        
        Returns:
        --------
        np.ndarray
            처리된 데이터
        """
        pass
    
    @abstractmethod
    def get_processor_info(self) -> Dict[str, Any]:
        """
        프로세서 정보 반환
        
        Returns:
        --------
        dict
            프로세서 메타데이터
        """
        pass




