import os
import ssl
import time
from contextlib import asynccontextmanager

import requests
import logging
from dotenv import load_dotenv

# 设置 httpx 和 huggingface_hub 日志级别，减少刷屏
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("huggingface_hub").setLevel(logging.WARNING)

from sentence_transformers import SentenceTransformer, CrossEncoder

from langchain_core.embeddings import Embeddings

from RAG.hdwj.ai import router

from fastapi import FastAPI
load_and_process_documents=None
get_reranker=None
get_embeddings=None
LocalEmbeddings=None
_global_reranker=None
emnn=None
async def aaa():
    global get_reranker, get_embeddings,emnn,_global_reranker
    global collection, _text_list, client
    _global_reranker = CrossEncoder('BAAI/bge-reranker-base')

    def get_file_path(filename):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        parent_dir = os.path.dirname(current_dir)
        return os.path.join(parent_dir, filename)

    class LocalEmbeddings(Embeddings):
        def __init__(self, model_name="shibing624/text2vec-base-chinese"):
            self.model = SentenceTransformer(model_name)

        def embed_documents(self, texts):
            vectors = self.model.encode(texts, normalize_embeddings=True)
            return vectors.tolist()

        def embed_query(self, text):
            vector = self.model.encode([text], normalize_embeddings=True)
            return vector[0].tolist()
    emnn=LocalEmbeddings()

    if collection is not None:
        return
    db_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")
    client = chromadb.PersistentClient(path=db_path)
    collection = client.get_or_create_collection(
        name="psychology_knowledge",
        metadata={"hnsw:space": "cosine"}
    )
    # 重组语义块之前是小块现在我们将小块合成大块，为了方便检索我们不直接存两个是因为栈内存
    if collection.count() > 0:
        logging.info("数据库已存在，从数据库重建语义块")
        all_data = collection.get()
        if all_data and all_data['documents']:
            documents = all_data['documents']
            metadatas = all_data['metadatas']

            text_map = {}
            max_id = -1
            for i, doc in enumerate(documents):
                meta = metadatas[i]
                if meta is None or not isinstance(meta, dict):
                    continue
                raw_pid = meta.get("_text_list_id", 0)

                try:
                    pid = int(raw_pid)
                except (TypeError, ValueError):
                    pid = 0
                max_id = max(max_id, pid)
                if pid not in text_map:
                    text_map[pid] = []
                text_map[pid].append(doc)

            sorted_pids = sorted(text_map.keys())
            _text_list = ["\n".join(text_map[pid]) for pid in sorted_pids]
            logging.info(f"已重建{len(_text_list)}个语义块")
        return

    file_path = get_file_path("心理咨询师知识库.docx")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"找不到文件: {file_path}")

    loader = Docx2txtLoader(file_path)
    documents = loader.load()
    print(f"加载文档完成，共{len(documents)}页")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", "?", "！", "。", " ", ""],
        length_function=len
    )
    splits = text_splitter.split_documents(documents)
    small_chunks = [doc.page_content.strip() for doc in splits if doc.page_content.strip()]
    print(f"规则分割完成，共{len(small_chunks)}个片段")
    # 语义切割


    text_splitter = SemanticChunker(
        emnn,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=85
    )
    result = text_splitter.split_documents(splits)
    _text_list = [doc.page_content.strip() for doc in result if doc.page_content.strip()]
    print(f"语义分割完成，共{len(_text_list)}个语义块")

    # 父子分块
    sub_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
    all_chunks = []
    chunk_ids = []
    for idx, text in enumerate(_text_list):
        sub_chunks = sub_splitter.split_text(text)
        for sub in sub_chunks:
            all_chunks.append(sub)
            chunk_ids.append(idx)

    print("正在生成向量...")
    vectors = emnn.embed_documents(all_chunks)
    ids = [f"chunk_{i}" for i in range(len(all_chunks))]
    metadatas = [{"_text_list_id": chunk_ids[i]} for i in range(len(all_chunks))]
    # 存入数据库
    collection.add(ids=ids, documents=all_chunks, metadatas=metadatas, embeddings=vectors)
    print("✅ 知识库加载完成！")
@asynccontextmanager
async def lifespan(app:FastAPI):
    logging.info("程序正在启动中，模型正在加载中")
    await aaa()
    yield
    logging.info("程序执行完毕")
app=FastAPI(lifespan=lifespan)
app.include_router(router)



try:
    from langchain_community.document_loaders import Docx2txtLoader
    from langchain_text_splitters import RecursiveCharacterTextSplitter#langchain中的分割类
    from langchain_experimental.text_splitter import SemanticChunker
    import chromadb
    from langchain_chroma import Chroma

    RAG_AVAILABLE = True

    collection = None
    _text_list = []
    _VECTOR_DIM = 768
    client = None
    reranked_results = None
    RAGAS_text = []




    def context_hanshu(query):
        from RAG.hdwj.gjzcx import BM25
        global client, reranked_results

        bm25 = BM25(_text_list)
        keyword_results = bm25.kh(query)
        keyword_rank = {item[0]: i + 1 for i, item in enumerate(keyword_results)}

        embeddings =emnn
        db = Chroma(client=client,
                    collection_name="psychology_knowledge",
                    embedding_function=embeddings)
        retriever = db.as_retriever(search_kwargs={"k": 20})
        vector_results = retriever.invoke(query)
        #去重防止多个小文本同时属于一个大文本导致混合检索不准

        parent_rank = {}
        for idx, doc in enumerate(vector_results):
            pid = doc.metadata.get("_text_list_id")
            if pid is not None:
                rank = idx + 1
                if pid not in parent_rank:
                    parent_rank[pid] = rank

        vector_rank = {pid: rank for pid, rank in parent_rank.items()}
        all_ids = set(keyword_rank) | set(vector_rank)

        final_scores = {}
        for pid in all_ids:
            k_score = keyword_rank.get(pid, len(set(keyword_rank))+1)
            v_score = vector_rank.get(pid, len(set(vector_rank))+1)
            final_scores[pid] = 1 / (60 + k_score) + 1 / (60 + v_score)

        top_ids = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        final_texts = [_text_list[pid] for pid, score in top_ids]

        # 重排序
        try:
            reranker = _global_reranker
            pairs = [(query, doc) for doc in final_texts]
            scores = reranker.predict(pairs)
            ranked = sorted(zip(final_texts, scores), key=lambda x: x[1], reverse=True)
            final_texts = [doc for doc, score in ranked[:3]]
        except Exception as e:
            logging.error(f"Rerank失败: {e}")

        Context = []
        retrieved_knowledge = []
        for text in final_texts:
            Context.append({"role": "system", "content": f"参考知识：\n{text}"})
            retrieved_knowledge.append({
                "score": 99.9,
                "content": text[:300] + "..." if len(text) > 300 else text
            })

        return Context, retrieved_knowledge, final_texts

except ImportError as e:
    RAG_AVAILABLE = False
    print(f"依赖导入失败: {e}")

    def context_hanshu(query):
        return [], [], []