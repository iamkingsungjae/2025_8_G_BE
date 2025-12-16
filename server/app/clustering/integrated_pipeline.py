"""
통합 클러스터링 파이프라인
필터, 프로세서, 알고리즘을 조합하여 사용
"""

from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np

# from .core.pipeline import DynamicClusteringPipeline  # KMeans 사용으로 인해 비활성화
from .filters.base import BaseFilter
from .filters.panel_filter import PanelFilter
from .processors.base import BaseProcessor
from .processors.vector_processor import VectorProcessor
from .algorithms.base import BaseClusteringAlgorithm


class IntegratedClusteringPipeline:
    """
    통합 클러스터링 파이프라인
    필터, 프로세서, 알고리즘을 자유롭게 교체 가능
    """
    
    def __init__(
        self,
        filter: Optional[BaseFilter] = None,
        processor: Optional[BaseProcessor] = None,
        algorithm: Optional[BaseClusteringAlgorithm] = None,
        use_dynamic_strategy: bool = True
    ):
        """
        Parameters:
        -----------
        filter : BaseFilter, optional
            필터 인스턴스 (None이면 PanelFilter 사용)
        processor : BaseProcessor, optional
            프로세서 인스턴스 (None이면 VectorProcessor 사용)
        algorithm : BaseClusteringAlgorithm, optional
            알고리즘 인스턴스 (None이면 동적 전략 사용)
        use_dynamic_strategy : bool
            동적 전략 사용 여부 (True면 DynamicClusteringPipeline 사용)
        """
        self.filter = filter or PanelFilter()
        self.processor = processor or VectorProcessor()
        self.algorithm = algorithm
        # DynamicClusteringPipeline은 KMeans를 사용하므로 비활성화 (HDBSCAN만 사용)
        self.use_dynamic_strategy = False
        
        # 동적 전략 파이프라인 (비활성화 - KMeans 사용)
        # self.dynamic_pipeline: Optional[DynamicClusteringPipeline] = None
        # if use_dynamic_strategy:
        #     self.dynamic_pipeline = DynamicClusteringPipeline()
        self.dynamic_pipeline = None
    
    def cluster(
        self,
        data: pd.DataFrame,
        filter_params: Optional[Dict[str, Any]] = None,
        processor_params: Optional[Dict[str, Any]] = None,
        algorithm_params: Optional[Dict[str, Any]] = None,
        verbose: bool = True
    ) -> Dict[str, Any]:
        """
        클러스터링 수행
        
        Parameters:
        -----------
        data : pd.DataFrame
            입력 데이터
        filter_params : dict, optional
            필터 파라미터
        processor_params : dict, optional
            프로세서 파라미터
        algorithm_params : dict, optional
            알고리즘 파라미터
        verbose : bool
            상세 로그 출력 여부
        
        Returns:
        --------
        dict
            클러스터링 결과
        """
        if self.use_dynamic_strategy and self.dynamic_pipeline:
            # 동적 전략 사용
            if verbose:
                print("[통합 파이프라인] 동적 전략 모드 사용")
            return self.dynamic_pipeline.fit(data, verbose=verbose)
        
        # 커스텀 파이프라인 사용
        if verbose:
            print("[통합 파이프라인] 커스텀 모드 사용")
            print(f"  필터: {type(self.filter).__name__}")
            print(f"  프로세서: {type(self.processor).__name__}")
            print(f"  알고리즘: {type(self.algorithm).__name__ if self.algorithm else '동적 선택'}")
        
        # Step 1: 필터링
        if filter_params:
            filtered_data = self.filter.filter(data, **filter_params)
        else:
            filtered_data = data.copy()
        
        if len(filtered_data) == 0:
            return {
                'success': False,
                'error': '필터링 후 데이터가 없습니다.',
                'data': filtered_data
            }
        
        # Step 2: 데이터 처리
        if processor_params:
            X = self.processor.process(filtered_data, **processor_params)
        else:
            X = self.processor.process(filtered_data)
        
        # Step 3: 알고리즘 선택 및 실행
        if self.use_dynamic_strategy:
            # 동적 전략 사용: DynamicClusteringPipeline 사용
            dynamic_result = self.dynamic_pipeline.fit(filtered_data, verbose=verbose)
            
            # 결과 변환 (DynamicClusteringPipeline 결과를 IntegratedClusteringPipeline 형식으로)
            if dynamic_result.get('success'):
                result_data = dynamic_result.get('data', filtered_data)
                labels = result_data['cluster'].values if 'cluster' in result_data.columns else None
                
                return {
                    'success': True,
                    'data': result_data,
                    'labels': labels,
                    'n_clusters': dynamic_result.get('optimal_k', 0),
                    'cluster_sizes': dynamic_result.get('cluster_sizes', {}),
                    'strategy': dynamic_result.get('strategy'),
                    'optimal_k': dynamic_result.get('optimal_k'),
                    'features': dynamic_result.get('features', []),
                    'k_scores': dynamic_result.get('k_scores', []),
                    'filter_info': self.filter.get_filter_info(),
                    'processor_info': self.processor.get_processor_info(),
                    'algorithm_info': {
                        'algorithm': 'dynamic_kmeans',
                        'features': dynamic_result.get('features', []),
                        'optimal_k': dynamic_result.get('optimal_k'),
                        'silhouette_score': dynamic_result.get('silhouette_score'),
                        'davies_bouldin_score': dynamic_result.get('davies_bouldin_score'),
                        'calinski_harabasz_score': dynamic_result.get('calinski_harabasz_score'),
                    }
                }
            else:
                # 프로파일링 모드
                return {
                    'success': False,
                    'strategy': dynamic_result.get('strategy', 'simple_profiling'),
                    'profile': dynamic_result.get('profile', {}),
                    'reason': dynamic_result.get('reason', ''),
                    'data': filtered_data,
                    'labels': None,
                    'n_clusters': 0,
                    'cluster_sizes': {},
                    'filter_info': self.filter.get_filter_info(),
                    'processor_info': self.processor.get_processor_info(),
                    'algorithm_info': {}
                }
        
        # 비동적 전략: HDBSCAN만 사용
        if self.algorithm is None:
            from .algorithms.hdbscan import HDBSCANAlgorithm
            algorithm = HDBSCANAlgorithm()
        else:
            algorithm = self.algorithm
        
        # 알고리즘 파라미터 적용
        if algorithm_params:
            labels = algorithm.fit_predict(X, **algorithm_params)
        else:
            labels = algorithm.fit_predict(X)
        
        # 결과 구성
        result_data = filtered_data.copy()
        result_data['cluster'] = labels
        
        # 클러스터 크기 계산
        import pandas as pd
        cluster_sizes = pd.Series(labels).value_counts().to_dict()
        # -1 (노이즈) 제외한 클러스터 수
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        return {
            'success': True,
            'data': result_data,
            'labels': labels,
            'n_clusters': n_clusters,
            'cluster_sizes': cluster_sizes,
            'filter_info': self.filter.get_filter_info(),
            'processor_info': self.processor.get_processor_info(),
            'algorithm_info': algorithm.get_algorithm_info()
        }
    
    def set_filter(self, filter: BaseFilter):
        """필터 교체"""
        self.filter = filter
    
    def set_processor(self, processor: BaseProcessor):
        """프로세서 교체"""
        self.processor = processor
    
    def set_algorithm(self, algorithm: BaseClusteringAlgorithm):
        """알고리즘 교체"""
        self.algorithm = algorithm
        self.use_dynamic_strategy = False

