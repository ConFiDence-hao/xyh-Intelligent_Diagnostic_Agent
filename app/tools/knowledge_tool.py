"""知识检索工具 - 从向量数据库中检索相关信息

智能路由：根据问题复杂度自动选择基础检索或高级检索
- 简单问题：使用 VectorSearchService（快速响应）
- 复杂问题：使用 EnhancedSearchService（重排序、查询重写等）
"""

from typing import List, Tuple
import re

from langchain_core.documents import Document
from langchain_core.tools import tool
from loguru import logger

from app.config import config
from app.services.vector_search_service import vector_search_service, SearchResult
from app.services.enhanced_search_service import enhanced_search_service


@tool(response_format="content_and_artifact")
def retrieve_knowledge(query: str) -> Tuple[str, List[Document]]:
    """从知识库中检索相关信息来回答问题
    
    智能判断问题复杂度，自动选择合适的检索策略：
    - 简单问题（问候、简短查询）→ 基础向量检索
    - 复杂问题（专业术语、多概念）→ 增强检索（重排序+查询重写）
    
    Args:
        query: 用户的问题或查询
        
    Returns:
        Tuple[str, List[Document]]: (格式化的上下文文本, 原始文档列表)
    """
    try:
        logger.info(f"知识检索工具被调用: query='{query}'")
        
        # 智能判断：选择检索策略
        use_enhanced = _should_use_enhanced_search(query)
        
        if use_enhanced:
            logger.info("检测到复杂问题，使用增强检索服务")
            search_results = enhanced_search_service.search(query)
        else:
            logger.info("检测到简单问题，使用基础向量检索")
            search_results = vector_search_service.search_similar_documents(
                query=query, 
                top_k=config.rag_top_k
            )
        
        if not search_results:
            logger.warning("未检索到相关文档")
            return "没有找到相关信息。", []
        
        # 将 SearchResult 转换为 LangChain Document
        docs = [
            Document(
                page_content=result.content,
                metadata=result.metadata
            )
            for result in search_results
        ]
        
        # 格式化文档为上下文
        context = format_docs(docs)
        
        logger.info(f"检索到 {len(docs)} 个相关文档")
        return context, docs
        
    except Exception as e:
        logger.error(f"知识检索工具调用失败: {e}")
        return f"检索知识时发生错误: {str(e)}", []


def _should_use_enhanced_search(query: str) -> bool:
    """
    智能判断是否应该使用增强检索
    
    判断标准：
    1. 问题长度：超过 20 个字符认为是复杂问题
    2. 关键词检测：包含诊断、分析、原因等专业词汇
    3. 多概念：包含多个技术术语或问号
    4. 特殊场景：AIOps 相关问题
    
    Args:
        query: 用户查询
        
    Returns:
        bool: True 使用增强检索，False 使用基础检索
    """
    # 规则1: 简单问候和短句 → 基础检索
    simple_greetings = ["你好", "您好", "谢谢", "再见", "help", "帮助"]
    if any(greeting in query.lower() for greeting in simple_greetings):
        return False
    
    # 规则2: 很短的问题（< 10 字符）→ 基础检索
    if len(query.strip()) < 10:
        return False
    
    # 规则3: 包含复杂关键词 → 增强检索
    complex_keywords = [
        "诊断", "分析", "原因", "为什么", "如何排查",
        "故障", "告警", "异常", "性能", "优化",
        "diagnose", "analyze", "troubleshoot", "debug"
    ]
    if any(keyword in query.lower() for keyword in complex_keywords):
        return True
    
    # 规则4: 包含多个技术术语（通过空格或标点分隔的词数）
    # 中文按字符数，英文按单词数
    if len(query) > 30:  # 长问题通常需要更精准的检索
        return True
    
    # 规则5: 包含多个问号或专业符号
    if query.count('?') > 1 or query.count('？') > 1:
        return True
    
    # 默认：中等长度问题使用基础检索（平衡速度和准确性）
    return False


def format_docs(docs: List[Document]) -> str:
    """
    格式化文档列表为上下文文本
    
    Args:
        docs: 文档列表
        
    Returns:
        str: 格式化的上下文文本
    """
    formatted_parts = []
    
    for i, doc in enumerate(docs, 1):
        # 提取元数据
        metadata = doc.metadata
        source = metadata.get("_file_name", "未知来源")
        
        # 提取标题信息 (如果有)
        headers = []
        for key in ["h1", "h2", "h3"]:
            if key in metadata and metadata[key]:
                headers.append(metadata[key])
        
        header_str = " > ".join(headers) if headers else ""
        
        # 构建格式化文本
        formatted = f"【参考资料 {i}】"
        if header_str:
            formatted += f"\n标题: {header_str}"
        formatted += f"\n来源: {source}"
        formatted += f"\n内容:\n{doc.page_content}\n"
        
        formatted_parts.append(formatted)
    
    return "\n".join(formatted_parts)
