"""Milvus 客户端工厂模块"""

from loguru import logger
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
    MilvusException,
)

from app.config import config


class MilvusClientManager:
    """Milvus 客户端管理器"""

    # 常量定义
    COLLECTION_NAME: str = "biz"
    VECTOR_DIM: int = 1024
    ID_MAX_LENGTH: int = 100
    CONTENT_MAX_LENGTH: int = 8000

    def __init__(self) -> None:
        """初始化 Milvus 客户端管理器"""
        self._collection: Collection | None = None

    def connect(self) -> Collection:
        """
        连接到 Milvus 服务器并初始化 collection

        Returns:
            Collection: Collection 实例
        """
        try:
            logger.info(f"正在连接到 Milvus: {config.milvus_host}:{config.milvus_port}")
            
            # 建立 default 连接
            connections.connect(
                alias="default",
                host=config.milvus_host,
                port=str(config.milvus_port),
                timeout=config.milvus_timeout / 1000,
            )

            logger.info("成功连接到 Milvus")

            # 检查并创建 collection
            if not self._collection_exists():
                logger.info(f"collection '{self.COLLECTION_NAME}' 不存在，正在创建...")
                self._create_collection()
                logger.info(f"成功创建 collection '{self.COLLECTION_NAME}'")
            else:
                logger.info(f"collection '{self.COLLECTION_NAME}' 已存在")
                self._collection = Collection(self.COLLECTION_NAME)
                
                # 检查向量维度是否匹配
                schema = self._collection.schema
                vector_field = None
                existing_dim = None
                for field in schema.fields:
                    if field.name == "vector":
                        vector_field = field
                        break
                
                if vector_field and hasattr(vector_field, 'params') and 'dim' in vector_field.params:
                    existing_dim = vector_field.params['dim']
                    if existing_dim != self.VECTOR_DIM:
                        logger.warning(
                            f"检测到向量维度不匹配！当前 collection 维度: {existing_dim}, 配置维度: {self.VECTOR_DIM}"
                        )
                        logger.info(f"正在删除旧 collection '{self.COLLECTION_NAME}'...")
                        utility.drop_collection(self.COLLECTION_NAME)
                        logger.info(f"正在重新创建 collection '{self.COLLECTION_NAME}'...")
                        self._create_collection()
                        logger.info(f"成功重新创建 collection，维度: {self.VECTOR_DIM}")
                    else:
                        logger.info(f"向量维度匹配: {self.VECTOR_DIM}")

            # 加载 collection
            self._load_collection()

            return self._collection

        except MilvusException as e:
            logger.error(f"Milvus 操作失败: {e}")
            self.close()
            raise RuntimeError(f"Milvus 操作失败: {e}") from e
        except Exception as e:
            logger.error(f"连接 Milvus 失败: {e}")
            self.close()
            raise RuntimeError(f"连接 Milvus 失败: {e}") from e

    def _collection_exists(self) -> bool:
        """检查 collection 是否存在"""
        return utility.has_collection(self.COLLECTION_NAME)

    def _create_collection(self) -> None:
        """创建 biz collection"""
        fields = [
            FieldSchema(
                name="id",
                dtype=DataType.VARCHAR,
                max_length=self.ID_MAX_LENGTH,
                is_primary=True,
            ),
            FieldSchema(
                name="vector",
                dtype=DataType.FLOAT_VECTOR,
                dim=self.VECTOR_DIM,
            ),
            FieldSchema(
                name="content",
                dtype=DataType.VARCHAR,
                max_length=self.CONTENT_MAX_LENGTH,
            ),
            FieldSchema(
                name="metadata",
                dtype=DataType.JSON,
            ),
        ]

        schema = CollectionSchema(
            fields=fields,
            description="Business knowledge collection",
            enable_dynamic_field=False,
        )

        self._collection = Collection(
            name=self.COLLECTION_NAME,
            schema=schema,
        )

        # 创建索引
        index_params = {
            "metric_type": "L2",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
        }
        self._collection.create_index(field_name="vector", index_params=index_params)
        
        logger.info("成功创建 collection 和索引")

    def _load_collection(self) -> None:
        """加载 collection 到内存"""
        if self._collection is None:
            self._collection = Collection(self.COLLECTION_NAME)

        try:
            load_state = utility.load_state(self.COLLECTION_NAME)
            state_name = getattr(load_state, "name", str(load_state))
            if state_name != "Loaded":
                self._collection.load()
                logger.info(f"成功加载 collection '{self.COLLECTION_NAME}'")
            else:
                logger.info(f"Collection '{self.COLLECTION_NAME}' 已加载")
        except AttributeError:
            try:
                self._collection.load()
                logger.info(f"成功加载 collection '{self.COLLECTION_NAME}'")
            except MilvusException as e:
                error_msg = str(e).lower()
                if "already loaded" not in error_msg:
                    raise

    def get_collection(self) -> Collection:
        """获取 collection 实例"""
        if self._collection is None:
            raise RuntimeError("Collection 未初始化，请先调用 connect()")
        return self._collection

    def health_check(self) -> bool:
        """健康检查"""
        try:
            return utility.has_collection(self.COLLECTION_NAME)
        except Exception as e:
            logger.error(f"Milvus 健康检查失败: {e}")
            return False

    def close(self) -> None:
        """关闭连接"""
        try:
            if self._collection is not None:
                self._collection.release()
                self._collection = None
        except Exception as e:
            logger.warning(f"释放 collection 失败: {e}")

        try:
            if connections.has_connection("default"):
                connections.disconnect("default")
        except Exception as e:
            logger.warning(f"断开连接失败: {e}")

        logger.info("已关闭 Milvus 连接")


# 全局单例
milvus_manager = MilvusClientManager()
