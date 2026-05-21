"""
AIOps 智能运维接口
"""

import json
from fastapi import APIRouter
from sse_starlette.sse import EventSourceResponse
from loguru import logger

from app.models.aiops import AIOpsRequest
from app.services.aiops_service import aiops_service
from app.tasks.diagnosis_tasks import run_diagnosis_task
from app.core.redis_client import redis_client
import json
from fastapi import HTTPException

router = APIRouter()#创建路由的容器，专门用来管理这一组接口


@router.post("/aiops")#智能诊断的接口
async def diagnose_stream(request: AIOpsRequest):
    """
    AIOps 故障诊断接口（流式 SSE）

    **功能说明：**
    - 支持两种诊断模式：系统诊断和问题诊断
    - 系统诊断：自动获取当前系统的活动告警并分析
    - 问题诊断：针对用户对话中的问题进行深度分析
    - 使用 Plan-Execute-Replan 模式进行智能诊断
    - 流式返回诊断过程和结果

    **SSE 事件类型：**

    1. `status` - 状态更新
       ```json
       {
         "type": "status",
         "stage": "fetching_alerts",
         "message": "正在获取系统告警信息..."
       }
       ```

    2. `plan` - 诊断计划制定完成
       ```json
       {
         "type": "plan",
         "stage": "plan_created",
         "message": "诊断计划已制定，共 6 个步骤",
         "target_alert": {...},
         "plan": ["步骤1: ...", "步骤2: ..."]
       }
       ```

    3. `step_complete` - 步骤执行完成
       ```json
       {
         "type": "step_complete",
         "stage": "step_executed",
         "message": "步骤执行完成 (2/6)",
         "current_step": "查询系统日志",
         "result_preview": "...",
         "remaining_steps": 4
       }
       ```

    4. `report` - 最终诊断报告
       ```json
       {
         "type": "report",
         "stage": "final_report",
         "message": "最终诊断报告已生成",
         "report": "# 故障诊断报告\\n...",
         "evidence": {...}
       }
       ```

    5. `complete` - 诊断完成
       ```json
       {
         "type": "complete",
         "stage": "diagnosis_complete",
         "message": "诊断流程完成",
         "diagnosis": {...}
       }
       ```

    6. `error` - 错误信息
       ```json
       {
         "type": "error",
         "stage": "error",
         "message": "诊断过程发生错误: ..."
       }
       ```

    **使用示例：**
    ```bash
    curl -X POST "http://localhost:9900/api/aiops" \\
      -H "Content-Type: application/json" \\
      -d '{"session_id": "session-123"}' \\
      --no-buffer
    ```

    **前端使用示例：**
    ```javascript
    const eventSource = new EventSource('/api/aiops');

    eventSource.onmessage = (event) => {
      const data = JSON.parse(event.data);

      if (data.type === 'plan') {
        console.log('诊断计划:', data.plan);
      } else if (data.type === 'step_complete') {
        console.log('步骤完成:', data.current_step);
      } else if (data.type === 'report') {
        console.log('最终报告:', data.report);
      } else if (data.type === 'complete') {
        console.log('诊断完成');
        eventSource.close();
      }
    };
    ```

    Args:
        request: AIOps 诊断请求

    Returns:
        SSE 事件流
    """
    session_id = request.session_id or "default"
    diagnosis_type = request.diagnosis_type or "system"
    question_context = request.question_context
    
    logger.info(f"[会话 {session_id}] 收到 AIOps 诊断请求（流式），类型: {diagnosis_type}")

    async def event_generator():#异步生成器函数
        try:
            async for event in aiops_service.diagnose(
                session_id=session_id,
                diagnosis_type=diagnosis_type,
                question_context=question_context
            ):  # 调用核心服务，获取异步生成器
                # 发送事件
                yield {
                    "event": "message",
                    "data": json.dumps(event, ensure_ascii=False)
                }

                # 如果是完成或错误事件，结束流
                if event.get("type") in ["complete", "error"]:
                    break

            logger.info(f"[会话 {session_id}] AIOps 诊断流式响应完成")

        except Exception as e:
            logger.error(f"[会话 {session_id}] AIOps 诊断流式响应异常: {e}", exc_info=True)
            yield {
                "event": "message",
                "data": json.dumps({
                    "type": "error",
                    "stage": "exception",
                    "message": f"诊断异常: {str(e)}"
                }, ensure_ascii=False)
            }

    return EventSourceResponse(event_generator())


@router.post("/aiops/async")
async def diagnose_async(request: AIOpsRequest):
    """
    AIOps 异步诊断接口：立即返回任务 ID，由后台 Worker 处理
    """
    if not redis_client:
        raise HTTPException(status_code=500, detail="Redis 服务不可用")

    session_id = request.session_id or "default"
    
    # 简单的缓存检查（可选）
    cache_key = f"diag_cache:{hash(session_id)}"
    cached = redis_client.get(cache_key)
    if cached:
        return {"status": "cached", "data": json.loads(cached)}

    # 发送任务到消息队列
    task = run_diagnosis_task.delay(session_id=session_id)
    
    return {
        "message": "诊断任务已启动",
        "task_id": task.id,
        "session_id": session_id
    }


@router.get("/aiops/result/{session_id}")
async def get_diagnosis_result(session_id: str):
    """
    查询异步任务结果
    """
    if not redis_client:
        raise HTTPException(status_code=500, detail="Redis 服务不可用")
        
    result_key = f"diag_result:{session_id}"
    result = redis_client.get(result_key)
    
    if not result:
        return {"status": "pending", "message": "任务正在处理中..."}
        
    return {"status": "completed", "data": json.loads(result)}
