"""检索优化技术对比测试"""

from app.services.enhanced_search_service import enhanced_search_service
from app.services.vector_search_service import vector_search_service


def test_basic_search():
    """测试基础向量检索"""
    print("\n" + "="*60)
    print("测试1: 基础向量检索")
    print("="*60)
    
    query = "CPU使用率高怎么办"
    results = vector_search_service.search_similar_documents(query, top_k=3)
    
    print(f"查询: {query}")
    print(f"返回 {len(results)} 个结果:")
    for i, result in enumerate(results, 1):
        print(f"\n结果 {i}:")
        print(f"  分数: {result.score:.4f}")
        print(f"  内容: {result.content[:100]}...")


def test_enhanced_search_with_reranker():
    """测试带重排序的增强检索"""
    print("\n" + "="*60)
    print("测试2: 增强检索（查询重写 + 重排序）")
    print("="*60)
    
    query = "CPU使用率高怎么办"
    results = enhanced_search_service.search(
        query=query,
        use_reranker=True,
        use_hyde=False,
        use_query_rewrite=True
    )
    
    print(f"查询: {query}")
    print(f"返回 {len(results)} 个结果:")
    for i, result in enumerate(results, 1):
        print(f"\n结果 {i}:")
        print(f"  内容: {result.content[:100]}...")


def test_hyde_search():
    """测试 HyDE 检索"""
    print("\n" + "="*60)
    print("测试3: HyDE 检索（假设性文档嵌入）")
    print("="*60)
    
    query = "内存泄漏"
    results = enhanced_search_service.search(
        query=query,
        use_reranker=True,
        use_hyde=True,
        use_query_rewrite=False
    )
    
    print(f"查询: {query}")
    print(f"返回 {len(results)} 个结果:")
    for i, result in enumerate(results, 1):
        print(f"\n结果 {i}:")
        print(f"  内容: {result.content[:100]}...")


def test_multi_query_search():
    """测试多路召回检索"""
    print("\n" + "="*60)
    print("测试4: 多路召回检索")
    print("="*60)
    
    query = "服务不可用"
    results = enhanced_search_service.multi_query_search(query, num_variants=3)
    
    print(f"查询: {query}")
    print(f"返回 {len(results)} 个结果:")
    for i, result in enumerate(results, 1):
        print(f"\n结果 {i}:")
        print(f"  内容: {result.content[:100]}...")


def compare_methods():
    """对比不同检索方法的效果"""
    print("\n" + "="*60)
    print("测试5: 不同检索方法对比")
    print("="*60)
    
    query = "数据库连接超时"
    
    print(f"\n查询: {query}\n")
    
    # 方法1: 基础检索
    print("方法1: 基础向量检索")
    results1 = vector_search_service.search_similar_documents(query, top_k=3)
    for i, r in enumerate(results1, 1):
        print(f"  {i}. [分数: {r.score:.4f}] {r.content[:80]}...")
    
    # 方法2: 查询重写 + 重排序
    print("\n方法2: 查询重写 + 重排序")
    results2 = enhanced_search_service.search(
        query=query,
        use_reranker=True,
        use_hyde=False,
        use_query_rewrite=True
    )
    for i, r in enumerate(results2, 1):
        print(f"  {i}. {r.content[:80]}...")
    
    # 方法3: HyDE + 重排序
    print("\n方法3: HyDE + 重排序")
    results3 = enhanced_search_service.search(
        query=query,
        use_reranker=True,
        use_hyde=True,
        use_query_rewrite=False
    )
    for i, r in enumerate(results3, 1):
        print(f"  {i}. {r.content[:80]}...")


if __name__ == "__main__":
    print("\n" + "="*60)
    print("检索优化技术测试")
    print("="*60)
    
    try:
        test_basic_search()
        test_enhanced_search_with_reranker()
        test_hyde_search()
        test_multi_query_search()
        compare_methods()
        
        print("\n" + "="*60)
        print("所有测试完成！")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
