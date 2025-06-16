import os
import torch
import hashlib
import shutil
from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.responses import JSONResponse, StreamingResponse
import uvicorn
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import asyncio
from typing import List, Dict, Any
from fastapi.middleware.cors import CORSMiddleware

MODEL_NAME = "deepseek-r1:7b"
DATA_PATH = "./documents/furina.txt"
CHROMA_DB_PATH = "./chroma_db"
EMBEDDINGS_CACHE = "./embeddings_cache"
EMBEDDING_MODEL = "GanymedeNil/text2vec-large-chinese"

def file_md5(filepath):
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def load_document(file_path):
    loader = TextLoader(file_path, encoding="utf-8")
    return loader.load()

def load_all_documents(folder_path):
    docs = []
    print("\n📚 正在遍历加载知识库文件夹: {}".format(folder_path))
    for root, _, files in os.walk(folder_path):
        for file in files:
            if file.lower().endswith('.txt'):
                file_path = os.path.join(root, file)
                print(f"  - 载入: {os.path.relpath(file_path, folder_path)}")
                loader = TextLoader(file_path, encoding="utf-8")
                docs.extend(loader.load())
    print(f"✅ 共加载文本块: {len(docs)}\n")
    return docs

def split_documents(documents, chunk_size=500, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "？", "！", " ", ""]
    )
    return text_splitter.split_documents(documents)

def init_embeddings():
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
        cache_folder=EMBEDDINGS_CACHE
    )

def folder_md5(folder_path):
    import hashlib
    md5 = hashlib.md5()
    file_count = 0
    for root, _, files in os.walk(folder_path):
        for file in sorted(files):
            if file.lower().endswith('.txt'):
                file_count += 1
                file_path = os.path.join(root, file)
                with open(file_path, "rb") as f:
                    while True:
                        chunk = f.read(4096)
                        if not chunk:
                            break
                        md5.update(chunk)
    print(f"🔑 文档总数: {file_count}，知识库MD5: {md5.hexdigest()}")
    return md5.hexdigest()

def create_or_load_vector_db(docs, embeddings, db_path, folder_path="./documents"):
    hash_path = os.path.join(db_path, "doc.md5")
    cur_md5 = folder_md5(folder_path)
    old_md5 = None
    if os.path.exists(hash_path):
        with open(hash_path, "r") as f:
            old_md5 = f.read().strip()
    if os.path.exists(db_path) and old_md5 == cur_md5:
        print("🟢 检测到知识库未变更，直接加载向量数据库。\n")
        return Chroma(persist_directory=db_path, embedding_function=embeddings)
    if os.path.exists(db_path):
        print("🟡 检测到知识库变更，正在重建向量数据库……")
        shutil.rmtree(db_path)
    print("🔵 正在创建新的向量数据库……")
    vector_db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=db_path
    )
    os.makedirs(db_path, exist_ok=True)
    with open(hash_path, "w") as f:
        f.write(cur_md5)
    print("✅ 向量数据库创建完成！\n")
    return vector_db

def init_llm(model_name=MODEL_NAME, base_url="http://localhost:11434"):
    return Ollama(
        model=model_name,
        base_url=base_url,
        temperature=0.0,
        verbose=False
    )

def create_rag_chain(llm, vector_db, prompt_template=None):
    if prompt_template is None:
        prompt_template = (
            "你是一个山东大学知识库的专家助手，必须严格根据提供的上下文信息回答问题。\n"
            "如果上下文中有相关信息，请简洁、准确地提取答案；如果没有，请明确告知用户无法回答。\n\n"
            "上下文: {context}\n问题: {question}\n\n回答:"
        )
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    retriever = vector_db.as_retriever(search_kwargs={"k": 5})
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

class ChatRequest(BaseModel):
    question: str

class Message(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = MODEL_NAME
    messages: List[Message]
    temperature: float = 0.0
    stream: bool = False
    max_tokens: int = 1024

# 初始化RAG链（只初始化一次，提升性能）
documents = load_all_documents("./documents")
docs = split_documents(documents)
embeddings = init_embeddings()
vector_db = create_or_load_vector_db(docs, embeddings, CHROMA_DB_PATH, folder_path="./documents")
llm = init_llm()
# 严格模式：只允许基于知识库回答
def strict_prompt():
    return (
        "你是一个山东大学知识库的专家助手，必须严格根据提供的上下文信息回答问题。\n"
        "如果上下文中有相关信息，请简洁、准确地提取答案；如果没有，请明确告知用户无法回答。\n\n"

        "上下文: {context}\n问题: {question}\n\n回答:"
    )
# 自由模式：允许AI自由发挥
def free_prompt():
    return (
        "你是一个山东大学知识库的专家助手。请优先根据提供的上下文信息回答问题，如果上下文没有相关信息，也可以根据你自己的知识和常识进行合理推测和补充。\n"
        "上下文: {context}\n问题: {question}\n\n回答:"
    )
strict_rag_chain = create_rag_chain(llm, vector_db, prompt_template=strict_prompt())
free_rag_chain = create_rag_chain(llm, vector_db, prompt_template=free_prompt())

@app.post("/chat")
async def chat(req: ChatRequest):
    question = req.question.strip()
    if not question:
        async def error_stream():
            yield "请输入有效的问题。"
        return StreamingResponse(error_stream(), media_type="text/plain")
    # 默认使用严格模式
    rag_chain = strict_rag_chain
    async def answer_stream():
        result = rag_chain({"query": question})
        answer = result['result']
        # 按段落或句子流式输出
        for chunk in answer.split("\n"):
            if chunk.strip():
                yield chunk + "\n"
                await asyncio.sleep(0.05)  # 模拟流式效果
    return StreamingResponse(answer_stream(), media_type="text/plain")

@app.post("/v1/chat/completions")
async def openai_chat_completions(req: ChatCompletionRequest):
    # 拼接所有messages为上下文
    context = ""
    for m in req.messages:
        if m.role == "system":
            context += f"[系统设定]{m.content}\n"
        elif m.role == "user":
            context += f"[用户]{m.content}\n"
        elif m.role == "assistant":
            context += f"[助手]{m.content}\n"
    # 取最后一条user为问题
    user_message = next((m for m in reversed(req.messages) if m.role == "user"), None)
    if not user_message:
        return JSONResponse(content={"error": {"message": "No user message provided."}}, status_code=400)
    question = user_message.content.strip()
    if not question:
        return JSONResponse(content={"error": {"message": "Empty question."}}, status_code=400)
    # 根据model参数选择链
    model_name = req.model or "deepseek-r1:7b"
    use_free = (model_name.endswith(":free") or model_name.endswith("-free"))
    rag_chain = free_rag_chain if use_free else strict_rag_chain
    def get_openai_response():
        result = rag_chain({"query": question, "context": context})
        answer = result['result']
        return {
            "id": "chatcmpl-xxx",
            "object": "chat.completion",
            "created": int(__import__('time').time()),
            "model": req.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": answer},
                    "finish_reason": "stop"
                }
            ]
        }
    if req.stream:
        async def stream_gen():
            result = rag_chain({"query": question, "context": context})
            answer = result['result']
            for chunk in answer.split("\n"):
                if chunk.strip():
                    data = {
                        "id": "chatcmpl-xxx",
                        "object": "chat.completion.chunk",
                        "created": int(__import__('time').time()),
                        "model": req.model,
                        "choices": [
                            {
                                "delta": {"role": "assistant", "content": chunk+"\n"},
                                "index": 0,
                                "finish_reason": None
                            }
                        ]
                    }
                    yield f"data: {__import__('json').dumps(data, ensure_ascii=False)}\n\n"
                    await asyncio.sleep(0.05)
            # 结束信号
            yield "data: [DONE]\n\n"
        return StreamingResponse(stream_gen(), media_type="text/event-stream")
    else:
        return JSONResponse(content=get_openai_response())

if __name__ == "__main__":
    print("\n================= Ollama RAG Server 启动 =================")
    print("API地址: http://0.0.0.0:8000  (本机可用 http://127.0.0.1:8000)")
    print("知识库目录: ./documents\n")
    uvicorn.run("main_server:app", host="0.0.0.0", port=8000, reload=False)
