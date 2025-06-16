# Ollama RAG 本地知识库问答系统

## 项目简介

本项目基于 FastAPI + LangChain + Ollama + Chroma + HuggingFace Embeddings，实现了一个本地化的中文知识库问答系统，支持流式响应、角色扮演、OpenAI API 兼容、多轮对话、网页前端对接等功能。

---

## 主要特性

- **知识库自动遍历**：自动加载 `./documents` 目录下所有 txt 文件，变动自动重建向量库。
- **OpenAI API 兼容**：支持 `/v1/chat/completions`，可直接对接 openai-sdk、网页、第三方工具。
- **流式响应**：支持 SSE 流式输出，前端可边生成边显示。
- **角色扮演**：支持 system prompt，AI 可扮演指定角色。
- **两种模式**：严格知识库模式与自由发挥模式（通过 model 字段切换）。
- **CORS 支持**：网页前端可直接跨域调用。

---

## 快速启动

1. 安装依赖

   ```bash
   pip install -r requirements.txt
   # 或手动安装 fastapi uvicorn langchain chromadb sentence-transformers huggingface_hub
   ```

2. 启动 Ollama 本地服务（需提前下载 deepseek-r1:7b 模型）

   ```bash
   ollama serve
   ollama pull deepseek-r1:7b
   ```

3. 启动本地知识库服务

   ```bash
   python main_server.py
   ```

   启动后访问 http://127.0.0.1:8000

4. 添加/修改知识库
   - 将 txt 文件放入 `./documents` 目录，重启服务自动生效。

---

## API 文档

### 1. /chat

- POST
- 请求体：`{"question": "你的问题"}`
- 响应：流式纯文本

### 2. /v1/chat/completions

- POST，兼容 OpenAI 格式
- 支持参数：model、messages、temperature、stream、max_tokens
- model 支持：
  - `deepseek-r1:7b`（严格知识库）
  - `deepseek-r1:7b:free`（自由发挥）
- 响应：OpenAI 标准格式或 SSE 流式

---

## 前端对接

- 推荐使用本项目自带的 `<ai-chat>` 组件或 test_openai_stream.html 进行流式体验。
- 也可用 openai-sdk、Postman、curl 等工具直接调用。

---

## 依赖环境

- Python 3.8+
- fastapi, uvicorn, langchain, chromadb, sentence-transformers, huggingface_hub, pydantic
- Ollama 本地服务

---

## .gitignore 说明

- 已自动忽略：
  - Python 缓存 (**pycache**/, \*.pyc)
  - VSCode 配置 (.vscode/)
  - 数据库和模型缓存 (chroma_db/, db/, embeddings_cache/)
  - 日志、系统文件等

---

## 常见问题

- **知识库变动不生效？**
  - 请确认所有 txt 文件都放在 documents 目录，重启服务。
- **模型下载慢/失败？**
  - 建议提前手动下载 HuggingFace 模型到 embeddings_cache。
- **API 跨域？**
  - 已内置 CORS 支持，前端可直接调用。

---

## 目录结构示例

```
ollama/
├── main_server.py
├── main.py
├── test_server.py
├── test_openai_api.py
├── ai-chatbox/
│   ├── chatbox.html
│   ├── chatboxdisplay.html
│   └── ...
├── documents/
│   ├── furina.txt
│   ├── Sdu.txt
│   └── ...
├── chroma_db/
│   └── ...
├── embeddings_cache/
│   └── models--GanymedeNil--text2vec-large-chinese/
└── ...
```

---

## 致谢

- [LangChain](https://github.com/langchain-ai/langchain)
- [Ollama](https://ollama.com/)
- [Chroma](https://www.trychroma.com/)
- [HuggingFace](https://huggingface.co/)

---

如有问题欢迎提 issue 或交流！
