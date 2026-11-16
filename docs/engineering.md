# 工程规范：Agent 与 RAG

## 运行环境

推荐 Python 3.11 或 3.12。`faiss-cpu`、`sentence-transformers` 在 3.13 上不一定有现成 wheel；如果必须用 3.13，先确认依赖能装再开工。

依赖分三层：

- 应用层：`fastapi`、`uvicorn`、`pydantic`。
- 编排层：`langgraph`、`langgraph-checkpoint-sqlite`、`langchain-openai`。
- 检索层：`langchain-huggingface`、`sentence-transformers`、`faiss-cpu`、`rank-bm25`、`jieba`。

## RAG 架构

1. 文档按 Markdown 标题切分：`MarkdownHeaderTextSplitter` 保留章节路径，再用 `RecursiveCharacterTextSplitter` 按 500 字、150 字重叠细分。
2. 向量化用 `BAAI/bge-small-zh-v1.5`，开启 `normalize_embeddings=True`；建索引和查询必须用同一个模型。
3. 向量库用 FAISS，元数据里写 chunk 的整数 `id`、`source` 和 `title_path`，避免用文本反查。
4. 查询时向量召回 top-20，BM25 召回 top-20，用 RRF 融合；需要更高精度时用 `BAAI/bge-reranker-base` 交叉编码器精排到 top-3。
5. rerank 模型下载失败或显存不足时，自动退回 RRF 排序，不要让整个请求挂掉。

## delete_user 工具的权限

`delete_user` 属于高危工具，权限必须在服务端做，不能靠提示词约束。

- 工具只接受目标用户 id，不接受“当前操作人”参数；当前操作人的 `user_id` 由服务端从 JWT 或会话里取出，通过闭包注入。
- 调用前检查当前用户的 `role` claim，只有 `admin` 才允许执行。
- 删除前写审计日志：操作人、目标用户、时间、请求 id。
- 工具返回统一错误码，不泄露内部表结构。
- 生产环境再加二次确认或审批流。

```python
# 反例：不要把 user_id 放进工具 schema，让模型自己填
def delete_user(user_id: str): ...

# 正例：user_id 由服务端闭包注入
def make_delete_tool(current_user_id: str, role: str):
    def delete_user(target_id: str):
        if role != "admin":
            return "permission denied"
        ...
```

## checkpointer.setup() 什么时候调用

`SqliteSaver.setup()` 用来建表，应该在服务启动或第一次创建 checkpointer 时调用一次，不要在每个请求、每个 graph 节点里调用。SQLite 连接要加 `check_same_thread=False`，否则 FastAPI 的线程池会报错。

```python
conn = sqlite3.connect("data/checkpoints.db", check_same_thread=False)
checkpointer = SqliteSaver(conn)
checkpointer.setup()
```

线程 id 不要只用前端传来的 `thread_id`。推荐拼成 `user_id:thread_id`，防止两个用户用同一个 thread_id 串会话。

## 流式接口

文章里的实现是 `text/plain` 的 chunked 流，不是 SSE。FastAPI 用 `StreamingResponse`，并加 `X-Accel-Buffering: no` 防止 Nginx 缓冲。前端用 `TextDecoder.decode(value, {stream: true})` 逐块解码，流结束时再 `decode()` 一次，把残留字节刷出来。

如果前端必须用 `EventSource`，再提供真正的 SSE 端点，事件格式为 `event: token` + `data: ...`。两种协议不要混在一个端点里。

## 结构化输出

DeepSeek 走 `with_structured_output` 时显式传 `method="function_calling"`。不传的话，LangChain 可能选 `json_schema`，DeepSeek 会返回 400。

校验失败时，把 `ValidationError` 和原始 prompt 一起喂回模型重试，最多 2 次；仍失败就抛错，不要返回半成品 JSON。

## 长期记忆

`InMemoryStore` 的 namespace 用 `(user_id,)`，写入 key 只允许字母、数字、下划线、点和短横线。`remember_fact` 和 `recall_facts` 的工具闭包里绑定 user_id，模型看不到也不需要传 user_id。

`InMemoryStore` 只适合演示，进程重启就丢。生产环境换 Redis、Postgres 或带持久化的 Store。

## 检索评测

`eval/retrieval_queries.json` 里每道题带 `expected` 关键词。`scripts/eval_retrieval.py` 对比向量 top-3、混合 top-3、混合+rerank top-3 的命中情况。先看命中率，再看延迟；不要只凭感觉调参数。
