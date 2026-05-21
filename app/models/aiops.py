"""
AIOps 请求和响应模型
"""

from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class AIOpsRequest(BaseModel):
    """AIOps 诊断请求"""
    
    session_id: Optional[str] = Field(
        default="default",
        description="会话ID，用于追踪诊断历史"
    )
    
    diagnosis_type: Optional[str] = Field(
        default="system",
        description="诊断类型：'system'（系统诊断）或 'question'（问题诊断）"
    )
    
    question_context: Optional[str] = Field(
        default=None,
        description="问题诊断时的对话上下文（仅当 diagnosis_type='question' 时使用）"
    )
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "session-123",
                "diagnosis_type": "system"
            }
        }


class AlertInfo(BaseModel):
    """告警信息"""
    alertname: str
    severity: str
    instance: str
    duration: str
    description: Optional[str] = None


class DiagnosisResponse(BaseModel):
    """诊断响应（非流式）"""
    
    code: int = 200
    message: str = "success"
    data: Dict[str, Any]
    
    class Config:
        json_schema_extra = {
            "example": {
                "code": 200,
                "message": "success",
                "data": {
                    "status": "completed",
                    "target_alert": {
                        "alertname": "HighCPUUsage",
                        "severity": "critical"
                    },
                    "diagnosis": {
                        "root_cause": "数据库连接池耗尽",
                        "recommendations": ["扩容数据库连接池", "优化SQL查询"]
                    }
                }
            }
        }
