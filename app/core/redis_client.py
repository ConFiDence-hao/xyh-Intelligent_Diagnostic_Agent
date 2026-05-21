"""Redis 客户端管理模块"""

import redis
from loguru import logger
from app.config import config

# 尝试从 config 获取配置，如果没有则使用默认值
REDIS_HOST = getattr(config, 'redis_host', 'localhost')
REDIS_PORT = getattr(config, 'redis_port', 6379)

# 创建全局 Redis 连接
try:
    redis_client = redis.Redis(
        host=REDIS_HOST, 
        port=REDIS_PORT, 
        db=0, 
        decode_responses=True,
        socket_connect_timeout=5,  # 设置连接超时时间
        socket_timeout=5  # 设置操作超时时间
    )
    redis_client.ping()
    logger.info("✅ Redis 连接成功")
except Exception as e:
    logger.warning(f"⚠️ Redis 连接失败: {e}。Redis 将不可用，但不影响其他功能。")
    logger.info("💡 提示: 如需使用 Redis，请运行: docker run -d --name redis -p 6379:6379 redis:latest")
    redis_client = None
