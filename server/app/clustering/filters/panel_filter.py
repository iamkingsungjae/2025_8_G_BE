"""
패널 필터 구현
검색 결과에 대한 필터링 로직
"""

from typing import Any, Dict, List, Optional
import pandas as pd
from .base import BaseFilter


class PanelFilter(BaseFilter):
    """패널 데이터 필터"""
    
    def __init__(self):
        """패널 필터 초기화"""
        self.applied_filters: Dict[str, Any] = {}
    
    def filter(
        self, 
        data: pd.DataFrame, 
        age_range: Optional[List[int]] = None,
        genders: Optional[List[str]] = None,
        regions: Optional[List[str]] = None,
        incomes: Optional[List[str]] = None,
        **kwargs
    ) -> pd.DataFrame:
        """
        패널 데이터 필터링
        
        Parameters:
        -----------
        data : pd.DataFrame
            필터링할 데이터
        age_range : List[int], optional
            [min_age, max_age]
        genders : List[str], optional
            성별 필터
        regions : List[str], optional
            지역 필터
        incomes : List[str], optional
            소득 필터
        **kwargs
            추가 필터 파라미터
        
        Returns:
        --------
        pd.DataFrame
            필터링된 데이터
        """
        filtered = data.copy()
        self.applied_filters = {}
        
        # 나이 필터
        if age_range and len(age_range) == 2:
            min_age, max_age = age_range
            if 'age' in filtered.columns:
                filtered = filtered[
                    (filtered['age'] >= min_age) & 
                    (filtered['age'] <= max_age)
                ]
                self.applied_filters['age_range'] = age_range
        
        # 성별 필터
        if genders:
            if 'gender' in filtered.columns:
                # 다양한 형식 지원 (M/F, 남/여, 남성/여성)
                gender_map = {
                    'M': ['M', '남', '남성', 'male', 'Male'],
                    'F': ['F', '여', '여성', 'female', 'Female']
                }
                valid_genders = []
                for g in genders:
                    if g in gender_map:
                        valid_genders.extend(gender_map[g])
                    else:
                        valid_genders.append(g)
                
                filtered = filtered[filtered['gender'].isin(valid_genders)]
                self.applied_filters['genders'] = genders
        
        # 지역 필터
        if regions:
            if 'location' in filtered.columns:
                filtered = filtered[filtered['location'].isin(regions)]
                self.applied_filters['regions'] = regions
        
        # 소득 필터
        if incomes:
            if 'Q6_label' in filtered.columns:
                filtered = filtered[filtered['Q6_label'].isin(incomes)]
                self.applied_filters['incomes'] = incomes
        
        # 추가 필터 (kwargs)
        for key, value in kwargs.items():
            if key in filtered.columns:
                if isinstance(value, list):
                    filtered = filtered[filtered[key].isin(value)]
                else:
                    filtered = filtered[filtered[key] == value]
                self.applied_filters[key] = value
        
        return filtered
    
    def get_filter_info(self) -> Dict[str, Any]:
        """필터 정보 반환"""
        return {
            'type': 'PanelFilter',
            'applied_filters': self.applied_filters.copy(),
            'filter_count': len(self.applied_filters)
        }




