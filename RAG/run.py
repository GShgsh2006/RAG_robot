import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import uvicorn

# 导入 ai 模块触发路由注册到 FastAPI app 上
import hdwj.ai

# 从 app.py 导入带 lifespan 的 FastAPI 实例
from hdwj.app import app

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000, log_level="info")
