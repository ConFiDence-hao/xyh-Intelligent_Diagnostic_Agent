"""向量存储管理器 - 封装 Milvus VectorStore 操作"""

from typing import List, Optional
from langchain_core.documents import Document
from loguru import logger

from app.config import config
from app.services.vector_embedding_service import vector_embedding_service
from app.core.milvus_client import milvus_manager


# 统一使用 biz collection
COLLECTION_NAME = "biz"


class VectorStoreManager:
    """向量存储管理器"""

    def __init__(self):
        """初始化向量存储管理器"""
        self.collection_name = COLLECTION_NAME
        self._initialized = False

    def _ensure_initialized(self):
        """确保 VectorStore 已初始化"""
        if not self._initialized:
            # 初始化 milvus_client（建立 default 连接）
            milvus_manager.connect()
            self._initialized = True

    def add_documents(self, documents: List[Document]) -> List[str]:
        """
        批量添加文档到向量存储（自动批量向量化）

        Args:
            documents: 文档列表

        Returns:
            List[str]: 文档 ID 列表
        """
        self._ensure_initialized()
        try:
            import time
            import uuid
            from pymilvus import Collection
            start_time = time.time()
            
            # 获取 collection
            collection: Collection = milvus_manager.get_collection()
            
            # 为每个文档生成唯一 id
            ids = [str(uuid.uuid4()) for _ in documents]
            
            # 批量生成向量
            texts = [doc.page_content for doc in documents]
            vectors = vector_embedding_service.embed_documents(texts)
            
            # 准备插入数据
            entities = []
            for i, doc in enumerate(documents):
                entity = {
                    "id": ids[i],
                    "vector": vectors[i],
                    "content": doc.page_content,
                    "metadata": doc.metadata or {},
                }
                entities.append(entity)
            
            # 插入数据
            collection.insert(entities)
            collection.flush()
            
            elapsed = time.time() - start_time
            logger.info(
                f"批量添加 {len(documents)} 个文档到 Milvus 完成, "
                f"耗时: {elapsed:.2f}秒, 平均: {elapsed/len(documents):.2f}秒/个"
            )
            return ids
        except Exception as e:
            logger.error(f"添加文档失败: {e}")
            raise

    def delete_by_source(self, file_path: str) -> int:
        """
        删除指定文件的所有文档

        Args:
            file_path: 文件路径

        Returns:
            int: 删除的文档数量
        """
        self._ensure_initialized()
        try:
            from pymilvus import Collection
            collection: Collection = milvus_manager.get_collection()
            
            # metadata 是 JSON 字段，使用 JSON 路径查询语法
            expr = f'metadata["_source"] == "{file_path}"'
            
            result = collection.delete(expr)
            deleted_count = result.delete_count if hasattr(result, "delete_count") else 0
            
            logger.info(f"删除文件旧数据: {file_path}, 删除数量: {deleted_count}")
            return deleted_count
            
        except Exception as e:
            logger.warning(f"删除旧数据失败 (可能是首次索引): {e}")
            return 0

    def similarity_search(self, query: str, k: int = 3) -> List[Document]:
        """
        相似度搜索

        Args:
            query: 查询文本
            k: 返回结果数量

        Returns:
            List[Document]: 相关文档列表
        """
        self._ensure_initialized()
        try:
            from pymilvus import Collection
            collection: Collection = milvus_manager.get_collection()
            
            # 生成查询向量
            query_vector = vector_embedding_service.embed_query(query)
            
            # 构建搜索参数
            search_params = {
                "metric_type": "L2",
                "params": {"nprobe": 10},
            }
            
            # 执行搜索
            results = collection.search(
                data=[query_vector],
                anns_field="vector",
                param=search_params,
                limit=k,
                output_fields=["id", "content", "metadata"],
            )
            
            # 解析搜索结果
            docs = []
            for hits in results:
                for hit in hits:
                    doc = Document(
                        page_content=hit.entity.get("content"),
                        metadata=hit.entity.get("metadata", {}),
                    )
                    docs.append(doc)
            
            logger.debug(f"相似度搜索完成: query='{query}', 结果数={len(docs)}")
            return docs
        except Exception as e:
            logger.error(f"相似度搜索失败: {e}")
            return []


# 全局变量（在 main.py 的 lifespan 中初始化）
_vector_store_manager: Optional["VectorStoreManager"] = None


def get_vector_store_manager() -> "VectorStoreManager":
    """获取 VectorStoreManager 单例"""
    if _vector_store_manager is None:
        raise RuntimeError("VectorStoreManager 未初始化，请确保应用已启动")
    return _vector_store_manager


def set_vector_store_manager(manager: "VectorStoreManager"):
    """设置 VectorStoreManager 单例（仅在 main.py 的 lifespan 中调用）"""
    global _vector_store_manager
    _vector_store_manager = manager
