import os
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import sentence_transformers
import hashlib
import shutil

# 设置模型和数据路径
MODEL_NAME = "deepseek-r1:7b"
DATA_PATH = "./documents/furina.txt"
CHROMA_DB_PATH = "./chroma_db"
EMBEDDINGS_CACHE = "./embeddings_cache"  # 嵌入模型缓存路径

# 1. 加载文档（增加文档内容打印，确认加载正确性）
def load_document(file_path):
    try:
        loader = TextLoader(file_path, encoding="utf-8")
        documents = loader.load()
        print(f"成功加载文档，共{len(documents)}个文档")
        if documents:
            print(f"文档预览: {documents[0].page_content[:200]}...")
        return documents
    except Exception as e:
        print(f"加载文档时出错: {e}")
        return []

# 2. 文本分割（减小chunk_size，提高检索精度）
def split_documents(documents, chunk_size=500, chunk_overlap=100):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "？", "！", " ", ""]
    )
    docs = text_splitter.split_documents(documents)
    print(f"文档分割完成，共{len(docs)}个文本块")
    # 打印前3个文本块预览
    for i, doc in enumerate(docs[:3]):
        print(f"文本块{i+1}预览: {doc.page_content[:100]}...")
    return docs

# 3. 初始化嵌入模型（明确指定模型参数）
def init_embeddings():
    try:
        # 显式指定模型路径，支持离线模式
        model_name = "GanymedeNil/text2vec-large-chinese"
        embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
            cache_folder=EMBEDDINGS_CACHE
        )
        print(f"嵌入模型初始化成功: {model_name}")
        return embeddings
    except Exception as e:
        print(f"初始化嵌入模型时出错: {e}")
        return None

# 4. 创建向量数据库
def create_vector_db(docs, embeddings, db_path):
    try:
        # 清空旧数据库（用于调试）
        if os.path.exists(db_path):
            import shutil
            shutil.rmtree(db_path)
            print("已清空旧向量数据库")
        
        print("创建新的向量数据库...")
        vector_db = Chroma.from_documents(
            documents=docs,
            embedding=embeddings,
            persist_directory=db_path
        )
        print("向量数据库创建完成")
        
        # 调试：检索测试文档
        test_query = "芙宁娜的生日"
        docs = vector_db.similarity_search(test_query, k=1)
        if docs:
            print(f"调试检索结果: {docs[0].page_content[:100]}...")
        else:
            print("调试检索失败，未找到相关文档")
        return vector_db
    except Exception as e:
        print(f"向量数据库操作出错: {e}")
        return None

def file_md5(filepath):
    with open(filepath, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def create_or_load_vector_db(docs, embeddings, db_path, doc_path=DATA_PATH):
    hash_path = os.path.join(db_path, "doc.md5")
    cur_md5 = file_md5(doc_path)
    old_md5 = None
    if os.path.exists(hash_path):
        with open(hash_path, "r") as f:
            old_md5 = f.read().strip()
    if os.path.exists(db_path) and old_md5 == cur_md5:
        print("加载已有向量数据库...")
        return Chroma(persist_directory=db_path, embedding_function=embeddings)
    if os.path.exists(db_path):
        shutil.rmtree(db_path)
    print("创建新的向量数据库...")
    vector_db = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=db_path
    )
    os.makedirs(db_path, exist_ok=True)
    with open(hash_path, "w") as f:
        f.write(cur_md5)
    return vector_db

# 5. 初始化LLM模型
def init_llm(model_name=MODEL_NAME, base_url="http://localhost:11434"):
    try:
        llm = Ollama(
            model=model_name,
            base_url=base_url,
            temperature=0.0,  # 设为0，确保回答确定性
            verbose=True
        )
        print(f"LLM模型初始化成功: {model_name}")
        return llm
    except Exception as e:
        print(f"初始化LLM模型时出错: {e}")
        return None

# 6. 创建RAG链（移除不兼容的search_type参数）
def create_rag_chain(llm, vector_db, prompt_template=None):
    try:
        # 优化提示模板，强调必须从上下文中获取信息
        if prompt_template is None:
            prompt_template = """
            你是一个芙宁娜知识库的专家助手，必须严格根据提供的上下文信息回答问题。
            如果上下文中有相关信息，请简洁、准确地提取答案；如果没有，请明确告知用户无法回答。

            上下文: {context}
            问题: {question}

            回答:
            """
        PROMPT = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        # 调整检索参数：移除search_type，仅使用k值控制检索数量
        retriever = vector_db.as_retriever(
            search_kwargs={"k": 5},
        )
        
        rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            chain_type_kwargs={"prompt": PROMPT},
            return_source_documents=True
        )
        print("RAG链创建完成")
        return rag_chain
    except Exception as e:
        print(f"创建RAG链时出错: {e}")
        return None

# 7. 执行问答测试（优化测试问题和输出）
def run_qa_test(rag_chain, test_questions):
    if rag_chain is None:
        print("无法执行测试，RAG链未正确初始化")
        return
    
    print("\n===== 开始芙宁娜知识库问答测试 =====")
    for i, question in enumerate(test_questions):
        print(f"\n问题 {i+1}: {question}")
        result = rag_chain({"query": question})
        print(f"回答: {result['result']}")
        
        # 详细显示来源文档
        if "source_documents" in result and result["source_documents"]:
            print("来源文档:")
            for j, doc in enumerate(result["source_documents"]):
                print(f"  文档片段{j+1}: {doc.page_content[:200]}...")
        else:
            print("来源: 无相关文档")
    print("\n===== 问答测试完成 =====")

# 8. 对话模式（与用户进行交互式问答）
def interactive_chat(rag_chain):
    print("\n===== 芙宁娜知识库对话模式 =====")
    print("输入你的问题，输入 exit 或 quit 退出。")
    while True:
        question = input("\n你：").strip()
        if question.lower() in ("exit", "quit"):
            print("对话结束。")
            break
        if not question:
            continue
        result = rag_chain({"query": question})
        print(f"助手：{result['result']}")
        if result.get("source_documents"):
            print("（参考文档片段）")
            for j, doc in enumerate(result["source_documents"]):
                print(f"  片段{j+1}: {doc.page_content[:100]}...")

# 主函数
def main():
    try:
        documents = load_document(DATA_PATH)
        if not documents:
            print("未加载到文档，程序终止")
            return
        docs = split_documents(documents)
        embeddings = init_embeddings()
        vector_db = create_or_load_vector_db(docs, embeddings, CHROMA_DB_PATH)
        llm = init_llm()
        rag_chain = create_rag_chain(llm, vector_db)
        interactive_chat(rag_chain)
    except Exception as e:
        print(f"程序运行出错: {e}")

if __name__ == "__main__":
    main()