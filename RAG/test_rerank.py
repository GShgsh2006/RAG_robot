"""
测试 BGE-Reranker 免费模型
"""
from sentence_transformers import CrossEncoder

# 1. 加载模型（第一次会自动下载，约 400MB）
print("正在加载模型...")
model = CrossEncoder('BAAI/bge-reranker-base')
print("模型加载完成！\n")

# 2. 测试数据
query = "压力大怎么办？"
documents = [
    "焦虑情绪的自我调节方法包括深呼吸、冥想和运动...",
    "抑郁症的诊断标准主要包括情绪低落、兴趣减退...",
    "缓解压力的有效方法有：规律作息、合理饮食、适度运动...",
]

# 3. 计算相关性分数
print(f"用户问题：{query}\n")
pairs = [(query, doc) for doc in documents]
scores = model.predict(pairs)

# 4. 按分数排序
results = list(zip(documents, scores))
results.sort(key=lambda x: x[1], reverse=True)

print("Rerank 结果（按相关性排序）：")
print("=" * 60)
for i, (doc, score) in enumerate(results, 1):
    print(f"排名 {i} | 分数：{score:.4f}")
    print(f"内容：{doc[:50]}...")
    print("-" * 60)

print("\n测试完成！模型可以正常使用。")
