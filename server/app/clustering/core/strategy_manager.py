"""
클러스터링 전략 결정 모듈
샘플 수에 따라 적절한 전략 선택
"""


def decide_clustering_strategy(n_samples):
    """
    샘플 수에 따른 클러스터링 전략 결정
    
    Parameters:
    -----------
    n_samples : int
        검색 결과 샘플 수
    
    Returns:
    --------
    dict : 전략 정보
        - strategy: 전략 이름
        - k_range: k 탐색 범위
        - min_cluster_size: 최소 클러스터 크기
        - reason: 전략 선택 이유
    """
    
    if n_samples < 100:
        return {
            'strategy': 'no_clustering',
            'k_range': None,
            'min_cluster_size': None,
            'reason': '샘플 수 부족 (< 100명)',
            'recommendation': '검색 조건 완화 또는 단순 프로파일링만 제공'
        }
    
    elif 100 <= n_samples < 500:
        # v3 기준으로 k 범위 축소
        max_k = min(4, n_samples // 50)  # 5 -> 4로 축소, 30 -> 50으로 증가
        min_k = 3
        if max_k < min_k:
            max_k = min_k
        return {
            'strategy': 'simple_clustering',
            'k_range': range(min_k, max_k + 1) if max_k >= min_k else range(min_k, min_k + 1),
            'min_cluster_size': max(30, n_samples // 20),
            'reason': '적은 샘플 (100~500명)',
            'recommendation': f'k={min_k}~{max_k}, 최소 클러스터 {max(30, n_samples // 20)}명'
        }
    
    elif 500 <= n_samples < 3000:
        # v3 기준으로 k 범위 축소
        max_k = min(5, n_samples // 100)  # 8 -> 5로 축소, 50 -> 100으로 증가
        min_k = 3
        if max_k < min_k:
            max_k = min_k
        return {
            'strategy': 'standard_clustering',
            'k_range': range(min_k, max_k + 1) if max_k >= min_k else range(min_k, min_k + 1),
            'min_cluster_size': max(30, n_samples // 20),
            'reason': '충분한 샘플 (500~3,000명)',
            'recommendation': f'k={min_k}~{max_k}, 표준 프로세스'
        }
    
    else:  # n_samples >= 3000
        # v3 기준: k=3이 최적이었으므로 k 범위를 더 보수적으로 설정
        max_k = min(6, n_samples // 500)  # 12 -> 6으로 축소, 100 -> 500으로 증가
        min_k = 3  # v3 최적값
        if max_k < min_k:
            max_k = min_k
        return {
            'strategy': 'advanced_clustering',
            'k_range': range(min_k, max_k + 1) if max_k >= min_k else range(min_k, min_k + 1),
            'min_cluster_size': max(50, n_samples // 40),
            'reason': '대용량 샘플 (3,000명 이상)',
            'recommendation': f'k={min_k}~{max_k}, 고급 분석 가능'
        }


def print_strategy_info(strategy):
    """전략 정보 출력"""
    print(f"\n{'='*60}")
    print(f"클러스터링 전략: {strategy['strategy']}")
    print(f"{'='*60}")
    print(f"이유: {strategy['reason']}")
    
    if strategy['k_range']:
        print(f"K 탐색 범위: {list(strategy['k_range'])}")
        print(f"최소 클러스터 크기: {strategy['min_cluster_size']}명")
    
    print(f"권장사항: {strategy['recommendation']}")
    print('='*60)


