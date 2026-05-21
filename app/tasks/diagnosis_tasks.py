"""Celery 异步任务模块 - 用于处理耗时的智能诊断任务"""

from celery import Celery
from app.services.aiops_service import aiops_service
from app.core.redis_client import redis_client
import json
from loguru import logger

# 使用 Redis 作为消息中间件 (broker) 和结果存储 (backend)
celery_app = Celery(
    'diagnosis_tasks',
    broker='redis://localhost:6379/1',
    backend='redis://localhost:6379/2'
)

@celery_app.task(bind=True, max_retries=3, default_retry_delay=60)
def run_diagnosis_task(self, user_input: str, session_id: str):
    """
    异步执行智能诊断任务
    """
    try:
        logger.info(f"🚀 Worker 开始处理任务: {session_id}")
        
        # 这里我们调用 service 的 execute 方法并收集最终结果
        final_report = ""
        for event in aiops_service.execute(user_input, session_id):
            if event.get("type") == "complete":
                final_report = event.get("response", "")
        
        # 将结果存入 Redis，设置 1 小时过期
        result_key = f"diag_result:{session_id}"
        redis_client.setex(result_key, 3600, json.dumps({
            "status": "success", 
            "report": final_report
        }))
        
        logger.info(f"✅ 任务完成: {session_id}")
        return {"status": "success"}
        
    except Exception as exc:
        logger.error(f"任务失败，准备重试: {exc}")
        raise self.retry(exc=exc)
