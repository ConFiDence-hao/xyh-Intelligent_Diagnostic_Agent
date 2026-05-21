"""测试多格式文档分割功能"""

from app.services.document_splitter_service import document_splitter_service

def test_markdown():
    """测试 Markdown 分割"""
    content = """# 标题1

这是第一段内容。

## 标题2

这是第二段内容，包含一些详细信息。

## 标题3

这是第三段内容。
"""
    docs = document_splitter_service.split_document(content, "test.md")
    print(f"✅ Markdown 分割: {len(docs)} 个分片")
    for i, doc in enumerate(docs):
        print(f"  分片 {i+1}: {len(doc.page_content)} 字符")

def test_python_code():
    """测试 Python 代码分割"""
    content = """
def hello_world():
    print("Hello, World!")

def calculate_sum(a, b):
    return a + b

class MyClass:
    def __init__(self):
        self.value = 0
    
    def increment(self):
        self.value += 1
"""
    docs = document_splitter_service.split_document(content, "test.py")
    print(f"✅ Python 代码分割: {len(docs)} 个分片")
    for i, doc in enumerate(docs):
        print(f"  分片 {i+1}: {len(doc.page_content)} 字符")

def test_json():
    """测试 JSON 分割"""
    content = """
{
    "database": {
        "host": "localhost",
        "port": 5432,
        "name": "mydb"
    },
    "server": {
        "host": "0.0.0.0",
        "port": 9000
    }
}
"""
    docs = document_splitter_service.split_document(content, "test.json")
    print(f"✅ JSON 分割: {len(docs)} 个分片")
    for i, doc in enumerate(docs):
        print(f"  分片 {i+1}: {doc.page_content[:100]}...")

def test_text():
    """测试普通文本分割"""
    content = "这是一段测试文本。" * 100
    docs = document_splitter_service.split_document(content, "test.txt")
    print(f"✅ 普通文本分割: {len(docs)} 个分片")

if __name__ == "__main__":
    print("=" * 50)
    print("测试多格式文档分割功能")
    print("=" * 50)
    
    test_markdown()
    print()
    
    test_python_code()
    print()
    
    test_json()
    print()
    
    test_text()
    print()
    
    print("=" * 50)
    print("所有测试完成！")
    print("=" * 50)
