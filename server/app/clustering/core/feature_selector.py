"""
동적 피쳐 선택 모듈
데이터 특성에 따라 유효한 피쳐만 자동 선택
"""

import pandas as pd
import numpy as np
from typing import List, Optional


class DynamicFeatureSelector:
    """동적 피쳐 선택기"""
    
    def __init__(self, 
                 min_variance=0.01,
                 max_missing_ratio=0.3,
                 max_imbalance_ratio=0.95,
                 candidate_features: Optional[List[str]] = None):
        """
        Parameters:
        -----------
        min_variance : float
            최소 분산 (이하면 제외)
        max_missing_ratio : float
            최대 결측 비율 (이상이면 제외)
        max_imbalance_ratio : float
            이진변수 최대 불균형 비율 (이상이면 제외)
        candidate_features : List[str], optional
            후보 피쳐 목록 (None이면 기본 목록 사용)
        """
        self.min_variance = min_variance
        self.max_missing_ratio = max_missing_ratio
        self.max_imbalance_ratio = max_imbalance_ratio
        
        # 기본 피쳐 풀 정의 (사용자가 커스터마이징 가능)
        # v3 최종 실험 기준: 6개 핵심 피쳐 우선, 추가 피쳐는 보조적으로 사용
        self.candidate_features = candidate_features or [
            # === v3 핵심 피쳐 (우선순위 1) ===
            'age_scaled',  # 나이 (v3 핵심)
            'Q6_scaled',  # 소득 (v3 핵심, 가장 중요)
            'education_level_scaled',  # 학력 (v3 핵심)
            'Q8_count_scaled',  # 전자제품 수 (v3 핵심)
            'Q8_premium_index',  # 프리미엄 지수 (v3 핵심)
            'is_premium_car',  # 프리미엄차 (v3 핵심)
            
            # === 보조 피쳐 (우선순위 2, v3에서 제거된 것들) ===
            # 이진변수는 v3에서 제거되었지만, 샘플이 적을 때는 유용할 수 있음
            # 'has_car',  # v3에서 제거 (너무 강력한 구분력)
            # 'has_smoking_experience',  # v3에서 제거 (이진변수 지배)
            
            # === 추가 연속형 피쳐 (우선순위 3) ===
            'age_z',  # Z-score 정규화된 연령 (age_scaled 대체 가능)
            
            # === 기타 이진 피쳐 (우선순위 4, 최소한만) ===
            # 'has_children',  # v2에서 제거 (다중공선성)
            # 'has_drinking_experience',
            # 'is_college_graduate',
            # 'is_employed',
            # 'is_student',
            # 'gender_M',
            # 'is_capital_area',
            # 'is_metropolitan',
        ]
    
    def select_features(self, df: pd.DataFrame, verbose: bool = True) -> Optional[List[str]]:
        """
        데이터프레임에서 유효한 피쳐 선택
        
        Parameters:
        -----------
        df : pd.DataFrame
            검색 결과 데이터
        verbose : bool
            상세 로그 출력 여부
        
        Returns:
        --------
        list : 선택된 피쳐 리스트 (None이면 선택 실패)
        """
        valid_features = []
        feature_stats = []
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"동적 피쳐 선택 (총 {len(self.candidate_features)}개 후보)")
            print('='*60)
        
        for feature in self.candidate_features:
            # 피쳐가 데이터에 없으면 스킵
            if feature not in df.columns:
                if verbose:
                    print(f"[경고] {feature}: 데이터에 없음 (스킵)")
                continue
            
            # 1. 결측치 체크
            missing_ratio = df[feature].isnull().sum() / len(df)
            if missing_ratio > self.max_missing_ratio:
                if verbose:
                    print(f"[제외] {feature}: 결측치 과다 ({missing_ratio:.1%})")
                continue
            
            # 2. 분산 체크
            variance = df[feature].var()
            if pd.isna(variance) or variance < self.min_variance:
                if verbose:
                    print(f"[제외] {feature}: 분산 너무 낮음 ({variance:.4f})")
                continue
            
            # 3. 이진변수 불균형 체크
            n_unique = df[feature].nunique()
            if n_unique == 2:
                value_counts = df[feature].value_counts(normalize=True)
                max_ratio = value_counts.max()
                if max_ratio > self.max_imbalance_ratio:
                    if verbose:
                        print(f"[제외] {feature}: 불균형 심함 ({max_ratio:.1%})")
                    continue
            
            # 유효한 피쳐 추가
            valid_features.append(feature)
            
            feature_stats.append({
                'feature': feature,
                'variance': variance,
                'missing_ratio': missing_ratio,
                'n_unique': n_unique
            })
            
            if verbose:
                print(f"[선택] {feature}: "
                      f"분산={variance:.4f}, "
                      f"결측={missing_ratio:.1%}, "
                      f"고유값={n_unique}")
        
        # 최소 피쳐 수 체크
        if len(valid_features) < 3:
            if verbose:
                print(f"\n[경고] 유효 피쳐 부족 ({len(valid_features)}개 < 3개)")
                print("→ 클러스터링 불가")
            return None
        
        if verbose:
            print(f"\n[완료] 최종 선택: {len(valid_features)}개 피쳐")
            print('='*60)
        
        return valid_features
    
    def get_feature_stats(self, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
        """선택된 피쳐의 통계 정보 반환"""
        stats = pd.DataFrame()
        
        for feature in features:
            stats = pd.concat([stats, pd.DataFrame({
                'feature': [feature],
                'mean': [df[feature].mean()],
                'std': [df[feature].std()],
                'min': [df[feature].min()],
                'max': [df[feature].max()],
                'variance': [df[feature].var()]
            })], ignore_index=True)
        
        return stats


