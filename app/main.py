"""FastAPI 应用入口

主应用程序，配置路由、中间件、静态文件等
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from contextlib import asynccontextmanager
import os
import asyncio

from app.config import config
from loguru import logger
from app.api import chat, health, file, aiops
from app.core.milvus_client import milvus_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 先初始化 VectorStore（建立 default 连接并创建 LangChain Milvus）
    from app.services.vector_store_manager import (
        VectorStoreManager,
        set_vector_store_manager,
    )
    manager = VectorStoreManager()
    manager._ensure_initialized()
    set_vector_store_manager(manager)
    
    # 再初始化 milvus_client（复用 default 连接，供其他模块使用）
    milvus_manager.connect()
    
    yield
    milvus_manager.close()


# 创建 FastAPI 应用
app = FastAPI(
    title=config.app_name,
    version=config.app_version,
    description="基于 LangChain 的智能oncall运维系统",
    lifespan=lifespan
)

# 配置 CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应该限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

#子文件里@router.get已经单条都注册路由了，include_router将子路由注册到主应用中
app.include_router(health.router, tags=["健康检查"])
app.include_router(chat.router, prefix="/api", tags=["对话"])
app.include_router(file.router, prefix="/api", tags=["文件管理"])
app.include_router(aiops.router, prefix="/api", tags=["AIOps智能运维"])

# 挂载静态文件
# 使用绝对路径，确保无论从哪个目录启动都能找到 static 文件夹
import sys
from pathlib import Path

# 获取项目根目录（app 的父目录）
project_root = Path(__file__).parent.parent
static_dir = project_root / "static"

if not static_dir.exists():
    raise RuntimeError(f"Static directory '{static_dir}' does not exist")

app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

@app.get("/")
async def root():
    """返回首页"""
    index_path = os.path.join(static_dir, "index.html")
    if os.path.exists(index_path):
        return FileResponse(index_path)
    return {
        "message": f"Welcome to {config.app_name} API",
        "version": config.app_version,
        "docs": "/docs"
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "app.main:app",
        host=config.host,
        port=config.port,
        reload=config.debug,
        log_level="info"
    )
