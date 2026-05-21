"""查询重写服务 - 优化用户查询提升检索效果"""

from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

from app.config import config
from app.core.llm_factory import LLMFactory


class QueryRewriterService:
    """查询重写服务"""

    def __init__(self):
        """初始化查询重写服务"""
        self.llm = LLMFactory.create_chat_model(
            model=config.rag_model,
            temperature=0.3  # 低温度，保证稳定性
        )
        self._create_prompt()
    
    def _create_prompt(self):
        """创建查询重写提示词模板"""
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的查询优化助手。请将用户的简短或不完整的查询重写为更详细、更适合检索的形式。

要求：
1. 保持原意不变
2. 补充可能的上下文
3. 使用专业术语
4. 长度控制在 50-100 字
5. 如果是技术问题，包含可能的关键词

示例：
- 输入："cpu 高了" → 输出："CPU使用率过高的原因分析和解决方案"
- 输入："内存泄漏" → 输出："内存泄漏的检测方法和修复步骤"
- 输入："服务挂了" → 输出："服务不可用的故障排查流程和恢复方法"""),
            ("human", "原始查询：{query}\n\n请重写为更适合检索的形式：")
        ])
    
    def rewrite_query(self, query: str) -> str:
        """
        重写用户查询
        
        Args:
            query: 原始查询
            
        Returns:
            str: 重写后的查询
        """
        try:
            logger.info(f"重写查询: {query[:50]}...")
            
            messages = self.prompt.format_messages(query=query)
            response = self.llm.invoke(messages)
            
            rewritten_query = response.content if hasattr(response, 'content') else str(response)
            
            # 清理结果（去除可能的引号或多余空格）
            rewritten_query = rewritten_query.strip().strip('"').strip("'")
            
            logger.info(f"查询重写完成: {rewritten_query[:80]}...")
            return rewritten_query
            
        except Exception as e:
            logger.error(f"查询重写失败: {e}")
            # 降级：返回原始查询
            return query
    
    def multi_query_rewrite(self, query: str, num_variants: int = 3) -> list:
        """
        生成多个查询变体（用于多路召回）
        
        Args:
            query: 原始查询
            num_variants: 变体数量
            
        Returns:
            list: 查询变体列表
        """
        try:
            logger.info(f"生成 {num_variants} 个查询变体 for: {query[:50]}...")
            
            prompt = ChatPromptTemplate.from_messages([
                ("system", f"""请将用户查询改写为 {num_variants} 个不同但语义相似的变体。

要求：
1. 每个变体从不同角度描述问题
2. 使用不同的关键词和表达方式
3. 保持专业性和准确性
4. 每行一个变体，不要编号

示例格式：
变体1
变体2
变体3"""),
                ("human", "原始查询：{query}")
            ])
            
            messages = prompt.format_messages(query=query)
            response = self.llm.invoke(messages)
            
            variants_text = response.content if hasattr(response, 'content') else str(response)
            
            # 解析变体（按行分割）
            variants = [v.strip() for v in variants_text.split('\n') if v.strip()]
            
            # 限制数量
            variants = variants[:num_variants]
            
            logger.info(f"生成 {len(variants)} 个查询变体")
            return variants
            
        except Exception as e:
            logger.error(f"多查询重写失败: {e}")
            # 降级：返回原始查询
            return [query]


# 全局单例
query_rewriter_service = QueryRewriterService()
