"""重排序服务 - 使用 Cross-Encoder 提升检索精度"""

from typing import List, Tuple

from loguru import logger
from sentence_transformers import CrossEncoder

from app.services.vector_search_service import SearchResult


class RerankerService:
    """重排序服务"""

    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        """
        初始化重排序器
        
        Args:
            model_name: 重排序模型名称
                - BAAI/bge-reranker-v2-m3 (推荐，支持多语言)
                - cross-encoder/ms-marco-MiniLM-L-6-v2 (英文专用，更快)
        """
        self.model_name = model_name
        self.reranker = None
        self._load_model()
    
    def _load_model(self):
        """加载重排序模型"""
        try:
            logger.info(f"正在加载重排序模型: {self.model_name}")
            self.reranker = CrossEncoder(self.model_name)
            logger.info("重排序模型加载成功")
        except Exception as e:
            logger.error(f"重排序模型加载失败: {e}")
            raise
    
    def rerank(
        self, 
        query: str, 
        results: List[SearchResult], 
        top_k: int = 3
    ) -> List[SearchResult]:
        """
        对检索结果进行重排序
        
        Args:
            query: 用户查询
            results: 初始检索结果（建议传入较多的候选，如 Top-50）
            top_k: 最终返回的数量
            
        Returns:
            List[SearchResult]: 重排序后的结果
        """
        if not results:
            return []
        
        try:
            # 1. 准备输入：(query, document) 对
            pairs = [(query, doc.content) for doc in results]
            
            # 2. 计算相关性分数
            scores = self.reranker.predict(pairs)
            
            # 3. 将分数附加到结果中
            scored_results = list(zip(results, scores))
            
            # 4. 按分数降序排序
            scored_results.sort(key=lambda x: x[1], reverse=True)
            
            # 5. 取 Top-K
            top_results = [result for result, score in scored_results[:top_k]]
            
            logger.info(
                f"重排序完成: {len(results)} 个候选 → {len(top_results)} 个结果, "
                f"最高分: {scored_results[0][1]:.4f}"
            )
            
            return top_results
            
        except Exception as e:
            logger.error(f"重排序失败: {e}")
            # 降级：返回原始结果
            return results[:top_k]


# 全局单例
reranker_service = RerankerService()
