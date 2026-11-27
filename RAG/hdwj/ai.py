import sys
import os
import json
import logging
from fastapi import Form, APIRouter
from fastapi.responses import StreamingResponse, HTMLResponse, FileResponse
import requests


from hdwj.Prompt import Faiss_Prompt

template_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'qdwj')

logging.basicConfig(level=logging.INFO)

OLLAMA_MODEL = "qwen3:1.7b"
OLLAMA_API_URL = "http://localhost:11434/api/chat"

# Ollama 流式调用
def ollama_chat_stream(messages):
    payload = {
        "model": OLLAMA_MODEL,
        "messages": messages,
        "stream": True,
        "think": True,
        "options": {
            "temperature": 0.4,
            "num_predict": 1024
        }
    }

    try:
        resp = requests.post(
            OLLAMA_API_URL,
            json=payload,
            stream=True,
            timeout=120
        )
        resp.raise_for_status()

        for line in resp.iter_lines():
            if not line:
                continue

            data = json.loads(line.decode("utf-8"))
            message = data.get("message", {})

            thinking = message.get("thinking", "")
            if thinking:
                yield "data: " + json.dumps(
                    {"type": "thinking", "content": thinking},
                    ensure_ascii=False
                ) + "\n\n"

            answer = message.get("content", "")
            if answer:
                yield "data: " + json.dumps(
                    {"type": "answer", "content": answer},
                    ensure_ascii=False
                ) + "\n\n"

            if data.get("done"):
                yield "data: " + json.dumps({"type": "done"}) + "\n\n"
                break

    except Exception as e:
        logging.error(f"Ollama 调用失败: {str(e)}")
        yield "data: " + json.dumps(
            {"type": "error", "content": str(e)},
            ensure_ascii=False
        ) + "\n\n"
router=APIRouter()
# ================= 首页 =================
@router.get('/')
async def index():
    return FileResponse(os.path.join(template_dir, 'index.html'))

# ================= AI 接口 =================
@router.post('/ai')
async def ai_Faiss(shuru: str = Form(...)):
    from hdwj.app import context_hanshu
    if not shuru:
        return HTMLResponse("请输入问题")

    try:
        logging.info(f"用户问题: {shuru}")

        Context, retrieved_knowledge, final_contexts = context_hanshu(shuru)

        Prompt = Faiss_Prompt.Prompt

        messages = []
        messages.append({"role": "system", "content": Prompt})
        messages.extend(Context)
        messages.append({"role": "user", "content": shuru})

        logging.info("开始流式生成...")
        return StreamingResponse(
            ollama_chat_stream(messages),
            media_type="text/event-stream"
        )

    except Exception as e:
        logging.error(f"AI 错误: {str(e)}", exc_info=True)
        return HTMLResponse(
            json.dumps({"type": "error", "content": str(e)}, ensure_ascii=False),
            status_code=500
        )

def main():
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=5000, log_level="info", reload=False)

if __name__ == "__main__":
    main()
