"""
필터 베이스 클래스
필터 기능을 확장 가능하게 만들기 위한 인터페이스
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import pandas as pd


class BaseFilter(ABC):
    """필터 베이스 클래스"""
    
    @abstractmethod
    def filter(self, data: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        데이터 필터링
        
        Parameters:
        -----------
        data : pd.DataFrame
            필터링할 데이터
        **kwargs
            필터 파라미터
        
        Returns:
        --------
        pd.DataFrame
            필터링된 데이터
        """
        pass
    
    @abstractmethod
    def get_filter_info(self) -> Dict[str, Any]:
        """
        필터 정보 반환
        
        Returns:
        --------
        dict
            필터 메타데이터
        """
        pass




