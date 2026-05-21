"""增强检索服务 - 整合重排序、HyDE、查询重写等技术"""

from typing import List

from loguru import logger

from app.config import config
from app.services.hyde_service import hyde_service
from app.services.query_rewriter_service import query_rewriter_service
from app.services.reranker_service import reranker_service
from app.services.vector_search_service import SearchResult, vector_search_service


class EnhancedSearchService:
    """增强检索服务"""

    def __init__(self):
        """初始化增强检索服务"""
        self.use_reranker = True       # 是否启用重排序
        self.use_hyde = False          # 是否启用 HyDE（默认关闭，因为较慢）
        self.use_query_rewrite = True  # 是否启用查询重写
        
        # 检索参数
        self.initial_top_k = 20        # 初始召回数量（重排序前）
        self.final_top_k = config.rag_top_k  # 最终返回数量
    
    def search(
        self, 
        query: str, 
        use_reranker: bool = None,
        use_hyde: bool = None,
        use_query_rewrite: bool = None
    ) -> List[SearchResult]:
        """
        执行增强检索
        
        Args:
            query: 用户查询
            use_reranker: 是否使用重排序（None 则使用默认配置）
            use_hyde: 是否使用 HyDE
            use_query_rewrite: 是否使用查询重写
            
        Returns:
            List[SearchResult]: 检索结果
        """
        # 使用传入参数或默认配置
        use_reranker = use_reranker if use_reranker is not None else self.use_reranker
        use_hyde = use_hyde if use_hyde is not None else self.use_hyde
        use_query_rewrite = use_query_rewrite if use_query_rewrite is not None else self.use_query_rewrite
        
        try:
            logger.info(f"开始增强检索: query='{query[:50]}...'")
            logger.info(f"配置: reranker={use_reranker}, hyde={use_hyde}, rewrite={use_query_rewrite}")
            
            current_query = query
            
            # 第1步：查询重写（可选）
            if use_query_rewrite:
                current_query = query_rewriter_service.rewrite_query(current_query)
            
            # 第2步：HyDE（可选）
            if use_hyde:
                results = hyde_service.hyde_search(
                    current_query, 
                    lambda q: self._vector_search(q, top_k=self.initial_top_k)
                )
            else:
                # 直接向量检索
                results = self._vector_search(current_query, top_k=self.initial_top_k)
            
            # 第3步：重排序（可选）
            if use_reranker and len(results) > self.final_top_k:
                results = reranker_service.rerank(
                    query=current_query,
                    results=results,
                    top_k=self.final_top_k
                )
            else:
                # 不需要重排序，直接截取 Top-K
                results = results[:self.final_top_k]
            
            logger.info(f"增强检索完成, 返回 {len(results)} 个结果")
            return results
            
        except Exception as e:
            logger.error(f"增强检索失败: {e}")
            # 降级：使用基础向量检索
            return self._vector_search(query, top_k=self.final_top_k)
    
    def _vector_search(self, query: str, top_k: int = 20) -> List[SearchResult]:
        """
        基础向量检索
        
        Args:
            query: 查询文本
            top_k: 返回数量
            
        Returns:
            List[SearchResult]: 检索结果
        """
        return vector_search_service.search_similar_documents(query, top_k=top_k)
    
    def multi_query_search(self, query: str, num_variants: int = 3) -> List[SearchResult]:
        """
        多路召回检索（生成多个查询变体，合并结果）
        
        Args:
            query: 原始查询
            num_variants: 变体数量
            
        Returns:
            List[SearchResult]: 去重后的检索结果
        """
        try:
            logger.info(f"开始多路召回检索: query='{query[:50]}...'")
            
            # 1. 生成多个查询变体
            variants = query_rewriter_service.multi_query_rewrite(query, num_variants)
            
            # 2. 对每个变体进行检索
            all_results = []
            for variant in variants:
                logger.info(f"检索变体: {variant[:60]}...")
                results = self._vector_search(variant, top_k=5)
                all_results.extend(results)
            
            # 3. 去重（基于文档 ID）
            seen_ids = set()
            unique_results = []
            for result in all_results:
                if result.id not in seen_ids:
                    seen_ids.add(result.id)
                    unique_results.append(result)
            
            # 4. 如果启用了重排序，对合并结果重排序
            if self.use_reranker and len(unique_results) > self.final_top_k:
                unique_results = reranker_service.rerank(
                    query=query,
                    results=unique_results,
                    top_k=self.final_top_k
                )
            else:
                unique_results = unique_results[:self.final_top_k]
            
            logger.info(f"多路召回完成, {len(all_results)} 个原始结果 → {len(unique_results)} 个去重结果")
            return unique_results
            
        except Exception as e:
            logger.error(f"多路召回检索失败: {e}")
            # 降级：单次检索
            return self._vector_search(query, top_k=self.final_top_k)


# 全局单例
enhanced_search_service = EnhancedSearchService()
