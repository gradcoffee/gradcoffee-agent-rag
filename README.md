# GradCoffee Agent RAG

项目一《RAG 知识库助手》的可运行代码。一个仓库把五篇文章里的能力串起来：

- 给 Agent 加一个 `search_docs` 工具；
- 用 LangGraph `Store` 做按用户隔离的长期记忆；
- 用 Pydantic + `with_structured_output` 输出稳定 JSON；
- 用 FastAPI 做 chunked 文本流，并附带真正的 SSE 端点；
- 向量 + BM25 双路召回、RRF 融合、bge-reranker 精排。

> JSON 相关工具推荐：[ForJSON](https://forjson.com/) —— 做 RAG 元数据、结构化输出时，用来格式化、校验和对比 JSON 很顺手。

## 对应文章

| 篇 | 文章 | 主要代码 |
| --- | --- | --- |
| 01 | [给它加一个 RAG 工具](https://gradcoffee.com/agent-specialty/project-rag/08-rag-tool/) | `config.py`、`chunking.py`、`indexer.py`、`tools.py` |
| 02 | [你的 Agent 每次都失忆：给它加个长期记忆](https://gradcoffee.com/agent-specialty/project-rag/09-long-term-memory/) | `memory.py`、`graph.py` |
| 03 | [让模型别再写小作文：用 Pydantic 把输出钉死](https://gradcoffee.com/agent-specialty/project-rag/10-structured-output/) | `schemas.py`、`structured.py` |
| 04 | [像 ChatGPT 一样打字：给你的 Agent 加流式输出](https://gradcoffee.com/agent-specialty/project-rag/11-streaming/) | `graph.py`、`api.py`、`web/index.html` |
| 05 | [你的 Agent 会查文档了，但查得准吗？——给 RAG 加 rerank 精排和混合检索](https://gradcoffee.com/agent-specialty/project-rag/14-hybrid-rerank/) | `fusion.py`、`retrieval.py`、`eval/` |

文章里的代码是分段讲的；这个仓库把它们拼成一个能跑的工程，命名和边界做了统一。

## 目录

```text
rag_agent/
  config.py       环境变量、模型名、检索参数
  fusion.py       纯函数 RRF
  chunking.py     Markdown 结构切块
  indexer.py      FAISS 建索引 + chunks.json
  retrieval.py    向量 / BM25 / RRF / rerank
  embeddings.py   bge-small-zh 工厂
  llm.py          DeepSeek 客户端
  memory.py       Store 长期记忆 + 两个工具
  tools.py        search_docs
  schemas.py      Pydantic DTO
  structured.py   结构化输出与校验重试
  graph.py        LangGraph + SQLite checkpointer
  api.py          FastAPI
scripts/          建索引、检索、问答、评测、结构化示例
docs/             示例知识库（员工手册 / 工程规范 / FAQ）
eval/             检索评测集
tests/            纯函数单测
web/              流式聊天前端
```

## 快速开始

推荐 Python 3.11 或 3.12。`faiss-cpu`、`sentence-transformers` 在 3.13 上不一定有现成 wheel，先在 3.11/3.12 里装依赖最稳。

Windows 上如果 `pip install faiss-cpu` 找不到 wheel，改用 conda 安装 `faiss-cpu`，或者在 WSL2 / Linux 容器里跑。代码本身是跨平台的，卡住的通常只是 FAISS 的二进制包。

```powershell
git clone <your-repo-url> gradcoffee-agent-rag
cd gradcoffee-agent-rag

python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

Copy-Item .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY
```

建索引：

```powershell
python -m scripts.ingest
```

第一次会下载 `BAAI/bge-small-zh-v1.5`（约 100MB）。后面检索时如果 `ENABLE_RERANK=true`，还会下载 `BAAI/bge-reranker-base`（约 1GB）。不想下载精排模型，就在 `.env` 里设：

```text
ENABLE_RERANK=false
```

只调试检索，不调 LLM：

```powershell
python -m scripts.search "一线城市差旅住宿标准是多少？"
python -m scripts.search "delete_user 工具怎么加权限？" --no-rerank --json
```

命令行问答：

```powershell
python -m scripts.ask "入职半年年假有几天？"
python -m scripts.ask "报销多久到账？" --no-stream
```

启动 API 和前端：

```powershell
uvicorn rag_agent.api:app --reload --port 8000
```

打开 <http://127.0.0.1:8000/>。前端走 `/api/chat`，是 `text/plain` 的 chunked 流；`/api/chat/sse` 是给 `EventSource` 用的真 SSE，方便对照两种协议。

## 关键实现

### 1. 检索工具

`search_docs` 在 `tools.py` 里，检索器懒加载。索引不存在时，工具返回“知识库检索失败：……”，而不是让服务启动就崩。模型看到的是可读错误，可以回复“先建索引”，不会直接 500。

切块用 `MarkdownHeaderTextSplitter` 保留章节路径，再用 `RecursiveCharacterTextSplitter` 按 500 字、150 字重叠细分。chunk 用整数 `id` 写进 FAISS metadata，检索结果直接按 id 回查，避免用文本反查。

### 2. 长期记忆

`memory.py` 里 `remember_fact` / `recall_facts` 的 `user_id` 来自闭包，不在工具 schema 里。模型只能填 `key` 和 `content`，没法给别的用户写记忆。namespace 是 `(user_id,)`，key 用正则限制为 `[A-Za-z0-9_.-]{1,64}`。

`InMemoryStore` 只适合演示，进程重启就丢。生产环境换成 Redis / Postgres Store。

### 3. 结构化输出

`structured.py` 显式传：

```python
get_llm().with_structured_output(AnswerWithSources, method="function_calling")
```

不传 `method` 时，LangChain 可能选 `json_schema`，DeepSeek 会返回 400。校验失败会把 `ValidationError` 和原始 prompt 一起喂回模型，最多重试 2 次。

```powershell
python -m scripts.structured_demo
```

### 4. 流式输出

`api.py` 的 `/api/chat`：

- `async for graph.astream(..., stream_mode="messages")`；
- `StreamingResponse`，`media_type="text/plain"`；
- `X-Accel-Buffering: no`，防止 Nginx 缓冲。

前端 `web/index.html` 用 `TextDecoder.decode(value, { stream: true })` 逐块解码，结束时再 `decode()` 一次 flush。中文多字节字符不会在块边界被截断。

### 5. 混合检索与 rerank

`retrieval.py`：

1. 向量召回 top-20；
2. jieba 分词后 BM25 召回 top-20；
3. `rrf_fuse` 融合；
4. `BAAI/bge-reranker-base` 交叉编码器精排到 top-3。

rerank 模型加载失败、下载失败或显存不足时，记录 warning 并退回 RRF 排序，请求不挂。

## 评测

```powershell
python -m scripts.eval_retrieval
python -m scripts.eval_retrieval --no-rerank
```

`eval/retrieval_queries.json` 每道题带 `expected` 关键词，脚本对比向量 top-3、混合 top-3、混合+rerank top-3 的命中。先看命中率，再看延迟，别凭感觉调参数。

## 测试

```powershell
pip install -r requirements-dev.txt
pytest
python -m compileall rag_agent scripts
```

`tests/` 里的用例只测不依赖 FAISS / 模型的纯函数和 Pydantic 校验。检索、LLM、FastAPI 需要装完依赖并配好 key 后手工跑。

## API

```powershell
# chunked 文本流
curl -N -X POST http://127.0.0.1:8000/api/chat ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"入职半年年假有几天？\",\"user_id\":\"demo_user\",\"thread_id\":\"default\"}"

# 结构化回答
curl -X POST http://127.0.0.1:8000/api/answer ^
  -H "Content-Type: application/json" ^
  -d "{\"question\":\"一线城市住宿标准是多少？\"}"

# 只看检索
curl -X POST http://127.0.0.1:8000/api/search ^
  -H "Content-Type: application/json" ^
  -d "{\"query\":\"delete_user 权限\",\"top_k\":3}"
```

## 生产化清单

- `user_id` 必须从 JWT / 会话取，不能信请求体。当前 API 为了演示允许传 `user_id`。
- `InMemoryStore` 换持久化 Store；SQLite checkpointer 换 Postgres，或者至少把 db 文件放到持久卷。
- FAISS 换 pgvector / Milvus 等支持并发写入和权限过滤的向量库。
- 给检索加文档 ACL：不同用户只能检索自己有权看的文档。
- 加限流、超时、重试和链路追踪；记录检索命中、rerank 分数、首 token 延迟。
- 工具返回的文档内容一律当“资料”，不当“指令”，防提示注入。
- reranker 放 GPU 或独立服务，别和 API 进程抢内存。

## 上传 GitHub

仓库当前没有初始化 `.git`，按需执行：

```powershell
git init
git add .
git commit -m "feat: RAG 知识库助手"
git branch -M main
git remote add origin https://github.com/<你的账号>/gradcoffee-agent-rag.git
git push -u origin main
```

`.env`、`data/index/`、`data/checkpoints.db` 已在 `.gitignore` 里，不会把密钥和运行时产物传上去。

## 关于作者

8 年 Java 后端（京东、字节）+ 2 年 AI 应用开发。公众号「梯度下降的咖啡师」，写 Agent 特调系列：从 RAG、Text2SQL 到多 Agent 编排，尽量把每个技术点落成能跑的代码。

下一篇会进入项目二：Text2SQL 数据查询助手，从“让产品经理自己查数据库”开始。

## License

MIT
