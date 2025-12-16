"""
통합 클러스터링 파이프라인
전체 프로세스를 통합 관리
"""

import pandas as pd
import numpy as np
# from sklearn.cluster import KMeans  # KMeans 제거됨 (HDBSCAN만 사용)
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from typing import Dict, Any, Optional

from .strategy_manager import decide_clustering_strategy, print_strategy_info
from .feature_selector import DynamicFeatureSelector
from .k_optimizer import DynamicKOptimizer


class DynamicClusteringPipeline:
    """동적 클러스터링 파이프라인"""
    
    def __init__(self, 
                 feature_selector: Optional[DynamicFeatureSelector] = None,
                 verbose: bool = True):
        """
        Parameters:
        -----------
        feature_selector : DynamicFeatureSelector, optional
            커스텀 피쳐 선택기 (None이면 기본 사용)
        verbose : bool
            상세 로그 출력 여부
        """
        self.feature_selector = feature_selector or DynamicFeatureSelector()
        self.k_optimizer: Optional[DynamicKOptimizer] = None
        self.strategy: Optional[Dict[str, Any]] = None
        self.selected_features: Optional[list] = None
        self.optimal_k: Optional[int] = None
        self.verbose = verbose
    
    def fit(self, df: pd.DataFrame, verbose: Optional[bool] = None) -> Dict[str, Any]:
        """
        동적 클러스터링 수행
        
        Parameters:
        -----------
        df : pd.DataFrame
            검색 결과 데이터
        verbose : bool, optional
            상세 로그 출력 여부 (None이면 초기화 시 설정값 사용)
        
        Returns:
        --------
        dict : 클러스터링 결과
        """
        if verbose is None:
            verbose = self.verbose
            
        n_samples = len(df)
        
        if verbose:
            print("\n" + "="*70)
            print(f"{'동적 클러스터링 파이프라인':^70}")
            print("="*70)
            print(f"검색 결과: {n_samples:,}명")
        
        # Step 1: 전략 결정
        self.strategy = decide_clustering_strategy(n_samples)
        
        if verbose:
            print_strategy_info(self.strategy)
        
        if self.strategy['strategy'] == 'no_clustering':
            if verbose:
                print(f"\n[경고] 샘플 수 부족 ({n_samples}개 < 100개)")
                print("→ 단순 프로파일링으로 대체\n")
            return {
                **self._simple_profiling(df),
                'reason': f'샘플 수 부족 ({n_samples}개 < 100개)',
                'sample_size': n_samples
            }
        
        # Step 2: 동적 피쳐 선택
        self.selected_features = self.feature_selector.select_features(df, verbose)
        
        if self.selected_features is None:
            if verbose:
                print("\n[오류] 유효 피쳐 부족으로 클러스터링 불가")
                print("→ 단순 프로파일링으로 대체\n")
            return {
                **self._simple_profiling(df),
                'reason': '유효 피쳐 부족 (3개 미만)',
                'sample_size': n_samples
            }
        
        # Step 3: 피쳐 매트릭스 생성
        X = df[self.selected_features].copy()
        
        # 결측치 처리 (혹시 모를 경우 대비)
        if X.isnull().sum().sum() > 0:
            if verbose:
                print("\n[경고] 결측치 발견, 평균값으로 대치")
            X = X.fillna(X.mean())
        
        # Step 4: 최적 k 탐색
        self.k_optimizer = DynamicKOptimizer(
            min_cluster_size=self.strategy['min_cluster_size']
        )
        
        k_result = self.k_optimizer.find_optimal_k(
            X, 
            self.strategy['k_range'],
            verbose
        )
        
        if k_result is None:
            if verbose:
                print("\n[오류] 최적 k를 찾을 수 없음")
                print("→ 단순 프로파일링으로 대체\n")
            return self._simple_profiling(df)
        
        self.optimal_k = k_result['optimal_k']
        
        # Step 5: 최종 클러스터링
        if verbose:
            print(f"\n{'='*60}")
            print("최종 클러스터링 수행")
            print('='*60)
        
        # KMeans 제거됨 (HDBSCAN만 사용)
        # DynamicClusteringPipeline은 더 이상 사용되지 않음
        # kmeans = KMeans(
        #     n_clusters=self.optimal_k, 
        #     random_state=42, 
        #     n_init=10, 
        #     max_iter=300
        # )
        # df = df.copy()  # 원본 데이터 보호
        # df['cluster'] = kmeans.fit_predict(X)
        
        # HDBSCAN 사용 (DynamicClusteringPipeline은 비활성화됨)
        from app.clustering.algorithms.hdbscan import HDBSCANAlgorithm
        hdbscan = HDBSCANAlgorithm()
        df = df.copy()
        df['cluster'] = hdbscan.fit_predict(X)
        
        # Step 6: 결과 평가
        final_silhouette = silhouette_score(X, df['cluster'])
        final_davies = davies_bouldin_score(X, df['cluster'])
        final_calinski = calinski_harabasz_score(X, df['cluster'])
        
        if verbose:
            print(f"Silhouette Score: {final_silhouette:.3f}")
            print(f"Davies-Bouldin Index: {final_davies:.3f}")
            print(f"Calinski-Harabasz Score: {final_calinski:.1f}")
            
            print(f"\n클러스터별 분포:")
            for cluster_id in range(self.optimal_k):
                count = (df['cluster'] == cluster_id).sum()
                pct = count / len(df) * 100
                print(f"  Cluster {cluster_id}: {count:,}명 ({pct:.1f}%)")
            
            print('='*60 + "\n")
        
        return {
            'success': True,
            'strategy': self.strategy['strategy'],
            'n_samples': n_samples,
            'n_features': len(self.selected_features),
            'features': self.selected_features,
            'optimal_k': self.optimal_k,
            'silhouette_score': final_silhouette,
            'davies_bouldin_score': final_davies,
            'calinski_harabasz_score': final_calinski,
            'cluster_sizes': df['cluster'].value_counts().to_dict(),
            'data': df,
            'k_scores': k_result['scores']
        }
    
    def _simple_profiling(self, df: pd.DataFrame) -> Dict[str, Any]:
        """클러스터링 불가능 시 단순 프로파일링"""
        if self.verbose:
            print("\n" + "="*60)
            print("단순 프로파일링 (클러스터링 대체)")
            print("="*60)
        
        profile: Dict[str, Any] = {
            'count': len(df),
        }
        
        # 연속형 변수
        if 'age' in df.columns:
            age_values = df['age'].dropna()
            if len(age_values) > 0:
                profile['age_mean'] = float(age_values.mean())
                profile['age_std'] = float(age_values.std())
                profile['age_min'] = float(age_values.min())
                profile['age_max'] = float(age_values.max())
                if self.verbose:
                    print(f"연령: {profile['age_mean']:.1f}세 (±{profile['age_std']:.1f})")
        
        # 성별 분포
        if 'gender' in df.columns:
            gender_dist = df['gender'].value_counts().to_dict()
            profile['gender_distribution'] = {str(k): int(v) for k, v in gender_dist.items()}
            if self.verbose:
                mode_gender = df['gender'].mode()[0] if len(df['gender'].mode()) > 0 else 'N/A'
                print(f"성별: {mode_gender}")
        elif 'gender_M' in df.columns:
            # 이진 변수인 경우
            male_count = int(df['gender_M'].sum())
            female_count = len(df) - male_count
            profile['gender_distribution'] = {'M': male_count, 'F': female_count}
            if self.verbose:
                print(f"성별: 남성 {male_count}명, 여성 {female_count}명")
        
        # 지역 분포
        if 'region_lvl1' in df.columns:
            region_dist = df['region_lvl1'].value_counts().to_dict()
            profile['region_lvl1_distribution'] = {str(k): int(v) for k, v in region_dist.items()}
            if self.verbose:
                mode_region = df['region_lvl1'].mode()[0] if len(df['region_lvl1'].mode()) > 0 else 'N/A'
                print(f"지역: {mode_region}")
        
        # 소득 정보
        if 'income_personal' in df.columns:
            income_values = df['income_personal'].dropna()
            if len(income_values) > 0:
                profile['income_personal_mean'] = float(income_values.mean())
                profile['income_personal_median'] = float(income_values.median())
                if self.verbose:
                    print(f"개인소득: 평균 {profile['income_personal_mean']:.0f}만원")
        elif 'income_household' in df.columns:
            income_values = df['income_household'].dropna()
            if len(income_values) > 0:
                profile['income_household_mean'] = float(income_values.mean())
                profile['income_household_median'] = float(income_values.median())
                if self.verbose:
                    print(f"가구소득: 평균 {profile['income_household_mean']:.0f}만원")
        
        # Q1 (결혼 상태) - qa_answers에서 파싱된 경우
        q1_cols = [col for col in df.columns if col.startswith('Q1')]
        if q1_cols:
            # Q1_미혼, Q1_기혼 같은 이진 변수들
            q1_info = {}
            for col in q1_cols:
                if df[col].dtype in ['int64', 'float64']:
                    count = int(df[col].sum())
                    if count > 0:
                        q1_info[col] = count
            if q1_info:
                profile['Q1_distribution'] = q1_info
        
        # Q4 (학력) - qa_answers에서 파싱된 경우
        q4_cols = [col for col in df.columns if col.startswith('Q4')]
        if q4_cols:
            q4_info = {}
            for col in q4_cols:
                if df[col].dtype in ['int64', 'float64']:
                    count = int(df[col].sum())
                    if count > 0:
                        q4_info[col] = count
            if q4_info:
                profile['Q4_distribution'] = q4_info
        
        # Q6 (소득) - qa_answers에서 파싱된 경우
        q6_cols = [col for col in df.columns if col.startswith('Q6')]
        if q6_cols:
            q6_info = {}
            for col in q6_cols:
                if df[col].dtype in ['int64', 'float64']:
                    count = int(df[col].sum())
                    if count > 0:
                        q6_info[col] = count
            if q6_info:
                profile['Q6_distribution'] = q6_info
        
        # 이진 변수들 요약
        binary_features = [
            'has_children', 'has_car', 'has_drinking_experience', 
            'has_smoking_experience', 'is_college_graduate',
            'is_employed', 'is_student'
        ]
        binary_summary = {}
        for feat in binary_features:
            if feat in df.columns and df[feat].dtype in ['int64', 'float64']:
                count = int(df[feat].sum())
                if count > 0:
                    binary_summary[feat] = {
                        'count': count,
                        'percentage': float(count / len(df) * 100)
                    }
        if binary_summary:
            profile['binary_features'] = binary_summary
        
        if self.verbose:
            print("="*60 + "\n")
        
        return {
            'success': False,
            'strategy': 'simple_profiling',
            'n_samples': len(df),
            'profile': profile,
            'data': df
        }
    
    def get_cluster_profiles(self, result: Dict[str, Any]) -> Optional[list]:
        """클러스터별 상세 프로파일 생성"""
        if not result.get('success'):
            return None
        
        df = result['data']
        profiles = []
        
        if self.verbose:
            print("\n" + "="*70)
            print(f"{'클러스터별 상세 프로파일':^70}")
            print("="*70)
        
        for cluster_id in range(result['optimal_k']):
            cluster_data = df[df['cluster'] == cluster_id]
            n = len(cluster_data)
            pct = n / len(df) * 100
            
            if self.verbose:
                print(f"\n{'='*70}")
                print(f"Cluster {cluster_id} (n={n:,}명, {pct:.1f}%)")
                print('='*70)
            
            profile: Dict[str, Any] = {
                'cluster_id': int(cluster_id),
                'size': int(n),
                'percentage': float(pct)
            }
            
            # 피쳐별 평균값
            if self.verbose:
                print("\n클러스터링 피쳐 평균:")
            for feature in result['features']:
                mean_val = float(cluster_data[feature].mean())
                profile[f'{feature}_mean'] = mean_val
                if self.verbose:
                    print(f"  {feature}: {mean_val:.3f}")
            
            # 추가 프로파일 정보
            if 'age' in cluster_data.columns:
                age_mean = float(cluster_data['age'].mean())
                age_std = float(cluster_data['age'].std())
                profile['age_mean'] = age_mean
                profile['age_std'] = age_std
                if self.verbose:
                    print(f"\n연령: {age_mean:.1f}세 (±{age_std:.1f})")
            
            # 범주형 변수 최빈값
            categorical_vars = ['age_group', 'generation', 'Q6_label', 
                               'Q4_label', 'Q1_label']
            
            for var in categorical_vars:
                if var in cluster_data.columns:
                    mode_val = cluster_data[var].mode()[0] if len(cluster_data[var].mode()) > 0 else 'N/A'
                    profile[var] = str(mode_val)
            
            # 일부 범주형 변수 출력
            if self.verbose:
                if 'age_group' in profile:
                    print(f"   연령대: {profile['age_group']}")
                if 'Q6_label' in profile:
                    print(f"소득: {profile['Q6_label']}")
                if 'Q4_label' in profile:
                    print(f"학력: {profile['Q4_label']}")
                if 'Q1_label' in profile:
                    print(f"결혼: {profile['Q1_label']}")
            
            profiles.append(profile)
        
        if self.verbose:
            print("\n" + "="*70 + "\n")
        
        return profiles

