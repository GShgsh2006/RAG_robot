import shutil
import os

db_path = "d:\\PythonProject4\\PythonProject\\znrwzs\\RAG\\hdwj\\chroma_db"
pkl_path = "d:\\PythonProject4\\PythonProject\\znrwzs\\RAG\\hdwj\\_text_list.pkl"

if os.path.exists(db_path):
    shutil.rmtree(db_path)
    print(f"已删除 {db_path}")
else:
    print(f"{db_path} 不存在")

if os.path.exists(pkl_path):
    os.remove(pkl_path)
    print(f"已删除 {pkl_path}")
else:
    print(f"{pkl_path} 不存在")