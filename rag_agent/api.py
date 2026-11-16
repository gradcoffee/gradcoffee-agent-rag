"""FastAPI：chunked 文本流 + 可选 SSE + 结构化回答。"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse

from . import config
from .graph import ainvoke_answer, stream_text
from .memory import delete_memory, list_memories
from .retrieval import get_retriever
from .schemas import ChatRequest, SearchRequest
from .structured import structured_answer

WEB_DIR = Path(__file__).resolve().parents[1] / "web"

app = FastAPI(
    title="GradCoffee Agent RAG",
    description="项目一《RAG 知识库助手》的可运行 API",
    version="0.1.0",
)

# 仅用于本地演示；生产环境请改成自己的前端域名。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _sse(event: str, data: str) -> str:
    """把一段文本包成 SSE 帧；JSON 编码避免换行破坏协议。"""
    payload = json.dumps(data, ensure_ascii=False)
    return f"event: {event}\ndata: {payload}\n\n"


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    return {
        "ok": True,
        "model": config.DEEPSEEK_MODEL,
        "embedding_model": config.EMBEDDING_MODEL,
        "reranker_model": config.RERANKER_MODEL,
        "rerank_enabled": config.ENABLE_RERANK,
        "index_exists": (config.INDEX_DIR / "chunks.json").exists(),
    }


@app.post("/api/chat")
async def chat(body: ChatRequest) -> StreamingResponse:
    """chunked 文本流，对应文章 04 的实现；不是 SSE。"""

    async def generate():
        try:
            async for token in stream_text(
                body.question,
                user_id=body.user_id,
                thread_id=body.thread_id,
            ):
                yield token
        except Exception as exc:
            yield f"\n[错误] {exc}\n"

    return StreamingResponse(
        generate(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/chat/sse")
async def chat_sse(body: ChatRequest) -> StreamingResponse:
    """真 SSE 版本，给需要 EventSource 的前端用。"""

    async def events():
        try:
            async for token in stream_text(
                body.question,
                user_id=body.user_id,
                thread_id=body.thread_id,
            ):
                yield _sse("token", token)
            yield _sse("done", "")
        except Exception as exc:
            yield _sse("error", str(exc))

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/answer")
def answer(body: ChatRequest) -> dict:
    """结构化回答：answer + sources + confidence。"""
    try:
        results = get_retriever().retrieve(body.question)
        result = structured_answer(body.question, results)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return result.model_dump()


@app.post("/api/search")
def search(body: SearchRequest) -> dict:
    try:
        results = get_retriever().retrieve(body.query, top_k=body.top_k)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"query": body.query, "results": [item.to_dict() for item in results]}


@app.post("/api/ask")
async def ask(body: ChatRequest) -> dict:
    """非流式问答，方便 curl 调试。"""
    try:
        text = await ainvoke_answer(
            body.question,
            user_id=body.user_id,
            thread_id=body.thread_id,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return {"answer": text}


@app.get("/api/memory")
def memory(
    user_id: str = Query(default="demo_user", min_length=1, max_length=128),
) -> dict:
    return {"user_id": user_id, "items": list_memories(user_id)}


@app.delete("/api/memory/{user_id}/{key}")
def forget(user_id: str, key: str) -> dict:
    try:
        delete_memory(user_id, key)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"ok": True, "user_id": user_id, "key": key}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("rag_agent.api:app", host="0.0.0.0", port=8000, reload=False)
