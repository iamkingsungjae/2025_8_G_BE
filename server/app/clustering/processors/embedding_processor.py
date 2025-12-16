"""
임베딩 프로세서 구현
텍스트/임베딩 데이터 처리
"""

from typing import Any, Dict, List, Optional
import numpy as np
import pandas as pd
from .base import BaseProcessor


class EmbeddingProcessor(BaseProcessor):
    """임베딩 데이터 프로세서"""
    
    def __init__(self, 
                 embedding_dim: int = 768,
                 normalize: bool = True):
        """
        Parameters:
        -----------
        embedding_dim : int
            임베딩 차원
        normalize : bool
            정규화 여부
        """
        self.embedding_dim = embedding_dim
        self.normalize = normalize
    
    def process(
        self, 
        data: pd.DataFrame, 
        embedding_column: str = 'embedding',
        **kwargs
    ) -> np.ndarray:
        """
        임베딩 데이터 처리
        
        Parameters:
        -----------
        data : pd.DataFrame
            처리할 데이터
        embedding_column : str
            임베딩 컬럼명
        **kwargs
            추가 파라미터
        
        Returns:
        --------
        np.ndarray
            처리된 임베딩 벡터
        """
        if embedding_column not in data.columns:
            raise ValueError(f"임베딩 컬럼 '{embedding_column}'이 없습니다.")
        
        # 임베딩 추출
        embeddings = []
        for idx, row in data.iterrows():
            emb = row[embedding_column]
            
            # 리스트/배열인 경우
            if isinstance(emb, (list, np.ndarray)):
                emb_array = np.array(emb, dtype=np.float32)
            # 문자열인 경우 (JSON 등)
            elif isinstance(emb, str):
                import json
                emb_array = np.array(json.loads(emb), dtype=np.float32)
            else:
                raise ValueError(f"임베딩 형식이 올바르지 않습니다: {type(emb)}")
            
            # 차원 확인
            if len(emb_array.shape) == 1:
                if emb_array.shape[0] != self.embedding_dim:
                    raise ValueError(
                        f"임베딩 차원 불일치: {emb_array.shape[0]} != {self.embedding_dim}"
                    )
                embeddings.append(emb_array)
            else:
                raise ValueError(f"임베딩이 1차원이 아닙니다: {emb_array.shape}")
        
        embeddings_array = np.array(embeddings, dtype=np.float32)
        
        # 정규화
        if self.normalize:
            norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
            norms[norms == 0] = 1  # 0으로 나누기 방지
            embeddings_array = embeddings_array / norms
        
        return embeddings_array
    
    def get_processor_info(self) -> Dict[str, Any]:
        """프로세서 정보 반환"""
        return {
            'type': 'EmbeddingProcessor',
            'embedding_dim': self.embedding_dim,
            'normalize': self.normalize
        }




