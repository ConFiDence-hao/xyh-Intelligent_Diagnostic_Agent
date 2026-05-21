"""文档分割服务模块 - 基于 LangChain 的智能文档分割"""

import json
import csv
import io
from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
    Language,
)
from loguru import logger

from app.config import config


class DocumentSplitterService:
    """文档分割服务 - 使用 LangChain 的分割器"""

    def __init__(self):
        """初始化文档分割服务"""
        self.chunk_size = config.chunk_max_size
        self.chunk_overlap = config.chunk_overlap

        # Markdown 标题分割器 (只按一级和二级标题分割，减少分片数)
        self.markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                # 不再按三级标题分割，避免过度碎片化
            ],
            strip_headers=False,  # 保留标题在内容中
        )

        # 递归字符分割器 (用于二次分割，使用更大的chunk_size)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size * 2,  # 加倍chunk_size，减少分片数
            chunk_overlap=self.chunk_overlap,
            length_function=len,
            is_separator_regex=False,
        )
        
        # 代码语言映射表
        self.language_map = {
            ".py": Language.PYTHON,
            ".js": Language.JS,
            ".java": Language.JAVA,
            ".go": Language.GO,
            ".cpp": Language.CPP,
            ".c": Language.C,
            ".ts": Language.TS,
        }
        
        # 格式分类常量
        self.MARKDOWN_FORMATS = [".md"]
        self.HTML_FORMATS = [".html", ".htm"]
        self.CODE_FORMATS = [".py", ".js", ".java", ".go", ".cpp", ".c", ".ts"]
        self.STRUCTURED_FORMATS = [".json", ".yaml", ".yml", ".csv", ".xml"]
        self.BINARY_FORMATS = [".pdf", ".docx"]

        logger.info(
            f"文档分割服务初始化完成, chunk_size={self.chunk_size}, "
            f"secondary_chunk_size={self.chunk_size * 2}, "
            f"overlap={self.chunk_overlap}"
        )

    def split_markdown(self, content: str, file_path: str = "") -> List[Document]:
        """
        分割 Markdown 文档 (两阶段分割 + 合并小片段)

        Args:
            content: Markdown 内容
            file_path: 文件路径 (用于元数据)

        Returns:
            List[Document]: 文档分片列表
        """
        if not content or not content.strip():
            logger.warning(f"Markdown 文档内容为空: {file_path}")
            return []

        try:
            # 第一阶段: 按标题分割
            md_docs = self.markdown_splitter.split_text(content)

            # 第二阶段: 按大小进一步分割
            docs_after_split = self.text_splitter.split_documents(md_docs)

            # 第三阶段: 合并太小的分片 (< 300字符)
            final_docs = self._merge_small_chunks(docs_after_split, min_size=300)

            # 添加文件路径元数据
            for doc in final_docs:
                doc.metadata["_source"] = file_path
                doc.metadata["_extension"] = ".md"
                doc.metadata["_file_name"] = Path(file_path).name

            logger.info(f"Markdown 分割完成: {file_path} -> {len(final_docs)} 个分片")
            return final_docs

        except Exception as e:
            logger.error(f"Markdown 分割失败: {file_path}, 错误: {e}")
            raise

    def split_text(self, content: str, file_path: str = "") -> List[Document]:
        """
        分割普通文本文档

        Args:
            content: 文本内容
            file_path: 文件路径 (用于元数据)

        Returns:
            List[Document]: 文档分片列表
        """
        if not content or not content.strip():
            logger.warning(f"文本文档内容为空: {file_path}")
            return []

        try:
            # 直接使用递归字符分割器
            docs = self.text_splitter.create_documents(
                texts=[content],
                metadatas=[
                    {
                        "_source": file_path,
                        "_extension": Path(file_path).suffix,
                        "_file_name": Path(file_path).name,
                    }
                ],
            )

            logger.info(f"文本分割完成: {file_path} -> {len(docs)} 个分片")
            return docs

        except Exception as e:
            logger.error(f"文本分割失败: {file_path}, 错误: {e}")
            raise

    def split_code(self, content: str, file_path: str = "") -> List[Document]:
        """
        分割代码文件（使用语言感知分割器）

        Args:
            content: 代码内容
            file_path: 文件路径

        Returns:
            List[Document]: 文档分片列表
        """
        if not content or not content.strip():
            logger.warning(f"代码文件内容为空: {file_path}")
            return []

        try:
            ext = Path(file_path).suffix.lower()
            language = self.language_map.get(ext)

            if language:
                # 使用语言专用分割器
                code_splitter = RecursiveCharacterTextSplitter.from_language(
                    language=language,
                    chunk_size=self.chunk_size * 2,
                    chunk_overlap=self.chunk_overlap,
                )
                docs = code_splitter.create_documents([content])
                logger.info(f"代码分割完成 ({language.value}): {file_path} -> {len(docs)} 个分片")
            else:
                # 不支持的语言，降级为普通文本分割
                docs = self.text_splitter.create_documents([content])
                logger.info(f"代码分割完成 (降级为文本): {file_path} -> {len(docs)} 个分片")

            # 添加元数据
            for doc in docs:
                doc.metadata["_source"] = file_path
                doc.metadata["_extension"] = ext
                doc.metadata["_file_name"] = Path(file_path).name

            return docs

        except Exception as e:
            logger.error(f"代码分割失败: {file_path}, 错误: {e}")
            raise

    def split_structured(self, content: str, file_path: str = "") -> List[Document]:
        """
        分割结构化数据文件（JSON/YAML/CSV）

        Args:
            content: 文件内容
            file_path: 文件路径

        Returns:
            List[Document]: 文档分片列表
        """
        if not content or not content.strip():
            logger.warning(f"结构化文件内容为空: {file_path}")
            return []

        try:
            ext = Path(file_path).suffix.lower()
            text_content = content

            # 根据格式转换结构化数据为自然语言
            if ext == ".json":
                text_content = self._json_to_text(content)
            elif ext in [".yaml", ".yml"]:
                text_content = self._yaml_to_text(content)
            elif ext == ".csv":
                text_content = self._csv_to_text(content)
            elif ext == ".xml":
                # XML 暂时按文本处理，后续可以添加专用解析
                pass

            # 使用文本分割器
            docs = self.text_splitter.create_documents(
                texts=[text_content],
                metadatas=[
                    {
                        "_source": file_path,
                        "_extension": ext,
                        "_file_name": Path(file_path).name,
                    }
                ],
            )

            logger.info(f"结构化数据分割完成: {file_path} -> {len(docs)} 个分片")
            return docs

        except Exception as e:
            logger.error(f"结构化数据分割失败: {file_path}, 错误: {e}")
            # 如果转换失败，降级为普通文本分割
            return self.split_text(content, file_path)

    def _json_to_text(self, json_content: str) -> str:
        """将 JSON 转换为自然语言描述"""
        try:
            data = json.loads(json_content)
            lines = []
            self._flatten_json(data, lines, prefix="")
            return "\n".join(lines)
        except Exception as e:
            logger.warning(f"JSON 解析失败，返回原始内容: {e}")
            return json_content

    def _flatten_json(self, data, lines: list, prefix: str = ""):
        """递归展平 JSON 结构"""
        if isinstance(data, dict):
            for key, value in data.items():
                new_prefix = f"{prefix}.{key}" if prefix else key
                if isinstance(value, (dict, list)):
                    self._flatten_json(value, lines, new_prefix)
                else:
                    lines.append(f"{new_prefix}: {value}")
        elif isinstance(data, list):
            for i, item in enumerate(data):
                new_prefix = f"{prefix}[{i}]"
                if isinstance(item, (dict, list)):
                    self._flatten_json(item, lines, new_prefix)
                else:
                    lines.append(f"{new_prefix}: {item}")
        else:
            lines.append(f"{prefix}: {data}")

    def _yaml_to_text(self, yaml_content: str) -> str:
        """将 YAML 转换为自然语言描述"""
        try:
            import yaml
            data = yaml.safe_load(yaml_content)
            lines = []
            self._flatten_json(data, lines, prefix="")
            return "\n".join(lines)
        except ImportError:
            logger.warning("PyYAML 未安装，返回原始内容")
            return yaml_content
        except Exception as e:
            logger.warning(f"YAML 解析失败，返回原始内容: {e}")
            return yaml_content

    def _csv_to_text(self, csv_content: str) -> str:
        """将 CSV 转换为自然语言描述"""
        try:
            reader = csv.reader(io.StringIO(csv_content))
            rows = list(reader)
            if not rows:
                return csv_content

            headers = rows[0]
            lines = []
            for row in rows[1:]:
                if len(row) == len(headers):
                    items = [f"{h}: {v}" for h, v in zip(headers, row)]
                    lines.append(", ".join(items))
                else:
                    lines.append(", ".join(row))
            return "\n".join(lines)
        except Exception as e:
            logger.warning(f"CSV 解析失败，返回原始内容: {e}")
            return csv_content

    def split_document(self, content: str, file_path: str = "") -> List[Document]:
        """
        智能分割文档 (根据文件类型选择分割器)

        Args:
            content: 文档内容
            file_path: 文件路径

        Returns:
            List[Document]: 文档分片列表
        """
        ext = Path(file_path).suffix.lower()
        
        # 根据格式选择分割策略
        if ext in self.MARKDOWN_FORMATS:
            return self.split_markdown(content, file_path)
        elif ext in self.CODE_FORMATS:
            return self.split_code(content, file_path)
        elif ext in self.STRUCTURED_FORMATS:
            return self.split_structured(content, file_path)
        elif ext in self.BINARY_FORMATS:
            # 二进制格式需要先解析，暂时给出提示
            logger.warning(
                f"二进制格式 {ext} 需要额外的解析库支持。"
                f"请安装 pypdf (PDF) 或 python-docx (DOCX) 后再使用。"
            )
            raise ValueError(
                f"暂不支持直接处理 {ext} 格式，请先转换为文本格式或安装相应的解析库"
            )
        else:
            # 默认使用文本分割器
            return self.split_text(content, file_path)

    def _merge_small_chunks(
        self, documents: List[Document], min_size: int = 300
    ) -> List[Document]:
        """
        合并太小的分片

        Args:
            documents: 文档列表
            min_size: 最小分片大小 (字符数)

        Returns:
            List[Document]: 合并后的文档列表
        """
        if not documents:
            return []

        merged_docs = []
        current_doc = None

        for doc in documents:
            doc_size = len(doc.page_content)

            if current_doc is None:
                # 第一个文档
                current_doc = doc
            elif doc_size < min_size and len(current_doc.page_content) < self.chunk_size * 2:
                # 当前文档太小且合并后不会太大，则合并
                current_doc.page_content += "\n\n" + doc.page_content
                # 保留主文档的元数据
            else:
                # 保存当前文档，开始新文档
                merged_docs.append(current_doc)
                current_doc = doc

        # 添加最后一个文档
        if current_doc is not None:
            merged_docs.append(current_doc)

        return merged_docs


# 全局单例
document_splitter_service = DocumentSplitterService()
