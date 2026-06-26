import numpy as np
import faiss
from typing import List
from reid.models.matcher.model import BaseMatcher
from reid.core.types import MatchResult

class FaissMatcher(BaseMatcher):
    """
    High-performance Matcher using FAISS.
    """
    def __init__(self, threshold: float = 0.7):
        super().__init__(threshold)
        self.index = None
        self.is_fitted = False
        # TODO faiss gpu 설정
        # GPU 사용 시 필요한 리소스 객체 (한 번만 할당해두는 것이 효율적입니다)
        # self.gpu_res = faiss.StandardGpuResources() if self.cfg.device == "cuda" else None

    def fit(self, embeddings: np.ndarray, labels: List[str]):
        if len(embeddings) == 0:
            return
            
        d = embeddings.shape[1]
        # Using IndexFlatIP for Cosine Similarity (assuming embeddings are L2 normalized)
        self.index = faiss.IndexFlatIP(d)

        # 1. 기본 CPU 인덱스 생성
        # cpu_index = faiss.IndexFlatIP(d)
        #
        # # 2. device 설정에 따라 인덱스를 GPU로 전송할지 결정
        # if self.cfg.device == "cuda":
        #     # 0번 GPU에 cpu_index를 올립니다.
        #     # 다중 GPU 환경이라면 device 번호를 파싱해서 두 번째 인자에 넣어줄 수 있습니다.
        #     self.index = faiss.index_cpu_to_gpu(self.gpu_res, 0, cpu_index)
        #     print("faiss loaded gpu")
        # else:
        #     # CPU 사용
        #     self.index = cpu_index
        #     print("faiss loaded cpu")
        
        # Ensure embeddings are float32 for Faiss
        embeddings_f32 = embeddings.astype('float32')
        self.index.add(embeddings_f32)
        
        self.labels = labels
        self.is_fitted = True

    def match(self, query_embedding: np.ndarray) -> MatchResult:
        if not self.is_fitted:
            return MatchResult(cat_id="Unknown", similarity=0.0, is_known=False)

        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
            
        query_f32 = query_embedding.astype('float32')
        
        # Search for top-1
        similarities, indices = self.index.search(query_f32, 1)
        
        best_idx = indices[0][0]
        similarity = similarities[0][0]
        
        is_known = similarity >= self.threshold
        cat_id = self.labels[best_idx] if is_known else "Unknown"
        
        return MatchResult(cat_id=cat_id, similarity=float(similarity), is_known=is_known)
