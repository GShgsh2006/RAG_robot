import sys
sys.path.insert(0, '.')
from hdwj.app import init_rag, context_hanshu
print("测试1: 导入成功")

init_rag()
print("测试2: 初始化成功")

Context, retrieved_knowledge, final_texts = context_hanshu("心理问题")
print(f"测试3: context_hanshu返回 {len(Context)} 个上下文")
print(f"  - 前200字: {final_texts[0][:200] if final_texts else '空'}")
print("全部测试通过!")