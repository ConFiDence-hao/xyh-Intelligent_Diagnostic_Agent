"""HyDE 服务 - 假设性文档嵌入提升检索效果"""

from langchain_core.prompts import ChatPromptTemplate
from loguru import logger

from app.config import config
from app.core.llm_factory import LLMFactory


class HyDEService:
    """HyDE (Hypothetical Document Embeddings) 服务"""

    def __init__(self):
        """初始化 HyDE 服务"""
        self.llm = LLMFactory.create_chat_model(
            model=config.rag_model,
            temperature=0.7  # 稍微高一点的温度，增加创造性
        )
        self._create_prompt()
    
    def _create_prompt(self):
        """创建 HyDE 提示词模板"""
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """你是一个专业的技术文档助手。请根据用户的问题，生成一个假设性的详细回答。

要求：
1. 回答要专业、准确、详细
2. 包含可能的解决方案、步骤或解释
3. 长度控制在 200-300 字
4. 不要说"我不知道"，尽量基于常识生成合理内容

这个假设性回答将用于检索相关文档，所以请确保内容与问题高度相关。"""),
            ("human", "问题：{question}")
        ])
    
    def generate_hypothetical_answer(self, question: str) -> str:
        """
        生成假设性答案
        
        Args:
            question: 用户问题
            
        Returns:
            str: 假设性答案
        """
        try:
            logger.info(f"生成假设性答案 for: {question[:50]}...")
            
            messages = self.prompt.format_messages(question=question)
            response = self.llm.invoke(messages)
            
            hypothetical_answer = response.content if hasattr(response, 'content') else str(response)
            
            logger.info(f"假设性答案生成完成, 长度: {len(hypothetical_answer)} 字符")
            return hypothetical_answer
            
        except Exception as e:
            logger.error(f"生成假设性答案失败: {e}")
            # 降级：返回原始问题
            return question
    
    def hyde_search(self, question: str, search_func) -> list:
        """
        执行 HyDE 检索流程
        
        Args:
            question: 用户问题
            search_func: 搜索函数，接受查询字符串，返回搜索结果列表
            
        Returns:
            list: 检索结果
        """
        try:
            # 1. 生成假设性答案
            hypothetical_answer = self.generate_hypothetical_answer(question)
            
            # 2. 用假设性答案进行检索
            logger.info("使用假设性答案进行检索...")
            results = search_func(hypothetical_answer)
            
            logger.info(f"HyDE 检索完成, 找到 {len(results)} 个结果")
            return results
            
        except Exception as e:
            logger.error(f"HyDE 检索失败: {e}")
            # 降级：直接用原问题检索
            return search_func(question)


# 全局单例
hyde_service = HyDEService()
