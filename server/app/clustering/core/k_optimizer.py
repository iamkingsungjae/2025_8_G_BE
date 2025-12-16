"""
최적 k 결정 모듈
샘플 수와 클러스터 품질을 고려한 동적 k 선택
"""

# from sklearn.cluster import KMeans  # KMeans 제거됨 (HDBSCAN만 사용)
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import numpy as np
import pandas as pd
from typing import Optional, Dict, Any


class DynamicKOptimizer:
    """동적 최적 k 결정기"""
    
    def __init__(self, min_cluster_size: int = 30):
        """
        Parameters:
        -----------
        min_cluster_size : int
            클러스터당 최소 샘플 수
        """
        self.min_cluster_size = min_cluster_size
    
    def find_optimal_k(
        self, 
        X: np.ndarray, 
        k_range: range, 
        verbose: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        최적 k 탐색
        
        Parameters:
        -----------
        X : array-like
            피쳐 매트릭스
        k_range : range
            탐색할 k 범위
        verbose : bool
            상세 로그 출력 여부
        
        Returns:
        --------
        dict : 최적 k 정보
            - optimal_k: 최적 k 값
            - scores: 모든 k별 점수
            - reason: 선택 이유
        """
        n_samples = len(X)
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"최적 K 탐색 (샘플 수: {n_samples}명)")
            print('='*60)
            print(f"K 범위: {list(k_range)}")
            print(f"최소 클러스터 크기: {self.min_cluster_size}명")
            print('-'*60)
            print(f"{'K':>3} | {'Silhouette':>11} | {'Davies-Bouldin':>15} | {'최소크기':>8}")
            print('-'*60)
        
        results = []
        valid_k_list = []
        
        for k in k_range:
            # k가 너무 커서 최소 클러스터 크기 위반 가능성 체크
            if n_samples // k < self.min_cluster_size:
                if verbose:
                    print(f"{k:>3} | [경고] 클러스터당 샘플 부족 (스킵)")
                continue
            
            # K-Means 제거됨 (HDBSCAN만 사용)
            # DynamicKOptimizer는 더 이상 사용되지 않음
            # kmeans = KMeans(n_clusters=k, random_state=42, n_init=10, max_iter=300)
            # labels = kmeans.fit_predict(X)
            
            # HDBSCAN 사용 (임시로 최적 k 찾기용)
            from app.clustering.algorithms.hdbscan import HDBSCANAlgorithm
            hdbscan = HDBSCANAlgorithm()
            labels = hdbscan.fit_predict(X)
            
            # 클러스터 크기 체크
            cluster_sizes = pd.Series(labels).value_counts()
            min_size = cluster_sizes.min()
            
            if min_size < self.min_cluster_size:
                if verbose:
                    print(f"{k:>3} | [경고] 최소 클러스터 크기 위반 ({min_size}명 < {self.min_cluster_size}명)")
                continue
            
            # 평가 지표 계산
            silhouette = silhouette_score(X, labels)
            davies_bouldin = davies_bouldin_score(X, labels)
            calinski = calinski_harabasz_score(X, labels)
            
            results.append({
                'k': k,
                'silhouette': silhouette,
                'davies_bouldin': davies_bouldin,
                'calinski_harabasz': calinski,
                'min_cluster_size': min_size
            })
            
            valid_k_list.append(k)
            
            if verbose:
                print(f"{k:>3} | {silhouette:>11.3f} | {davies_bouldin:>15.3f} | {min_size:>8}명")
        
        if not results:
            if verbose:
                print("\n[오류] 유효한 k를 찾을 수 없음")
            return None
        
        # 최적 k 결정 (Silhouette Score 기준)
        best_idx = np.argmax([r['silhouette'] for r in results])
        optimal_k = results[best_idx]['k']
        
        if verbose:
            print('-'*60)
            print(f"\n[완료] 최적 K: {optimal_k}")
            print(f"   Silhouette Score: {results[best_idx]['silhouette']:.3f}")
            print(f"   Davies-Bouldin Index: {results[best_idx]['davies_bouldin']:.3f}")
            print(f"   최소 클러스터 크기: {results[best_idx]['min_cluster_size']}명")
            print('='*60)
        
        return {
            'optimal_k': optimal_k,
            'scores': results,
            'reason': f"Silhouette Score 기준 최적 (k={optimal_k})",
            'best_scores': results[best_idx]
        }




