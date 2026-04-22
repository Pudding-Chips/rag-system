# =========================
# retriever.py
# =========================
import json
import chromadb
from db import get_client  # 复用 db.py 里的客户端配置
from embed import embed_texts

# 定义 Document 类，确保 api.py 和 reranker.py 可以通过 .page_content 访问内容
class SimpleDocument:
    def __init__(self, content, metadata):
        self.page_content = content  # 对应知识库里的 question
        self.metadata = metadata      # 包含 answer 和其他元数据

def retrieve(query, model="text2vec", biz="cdn", n_results=5):
    # 1. 直接连接到你 fill.py 成功的那个集合
    client = get_client()
    
    # 【关键修改】显式指定名称，对齐你 fill.py 里的 collection_name
    target_collection_name = "cdn_qa_v1_768" 
    
    try:
        collection = client.get_collection(name=target_collection_name)
    except Exception as e:
        print(f"[ERROR] 无法获取集合 {target_collection_name}，请检查 fill.py 是否运行成功。错误: {e}")
        return []
    
    # 2. 获取并处理查询向量
    query_embs = embed_texts(query, model)
    
    # 维度与格式纠偏
    if hasattr(query_embs, "tolist"):
        query_list = query_embs.tolist()
    else:
        query_list = list(query_embs)

    # 包装成双层列表 [[...]]
    if len(query_list) > 0 and not isinstance(query_list[0], list):
        final_emb = [query_list]
    else:
        final_emb = query_list

    # 3. 执行检索
    results = collection.query(
        query_embeddings=final_emb,
        n_results=n_results,
        include=["documents", "metadatas"]
    )
    
    # 4. 提取数据
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    
    print(f"[DEBUG] 目标集合: {target_collection_name} | 召回数量: {len(docs)}")

    # 5. 封装返回
    all_docs = []
    for i in range(len(docs)):
        doc_obj = SimpleDocument(
            content=docs[i],
            metadata=metas[i] if metas else {}
        )
        all_docs.append(doc_obj)
        
    return all_docs

# from db import get_collection
# from embed import embed_texts

# OLLAMA_URL = "http://localhost:11434/api/generate"
# LLM_MODEL = "llama3"

# def retrieve(query, model="text2vec", biz="cdn", n_results=5):
#     collection = get_collection(biz, model)
#     all_docs = []
#     query_emb = embed_texts(query, model)[0]
#     results = collection.query(
#         query_embeddings=[query_emb.tolist()],
#         n_results=n_results,
#         include=["documents", "metadatas", "distances"]
#     )
#     docs = results.get("documents", [[]])[0]
#     metas = results.get("metadatas", [[]])[0]
#     distances = results.get("distances", [[]])[0]
#     for i, d in enumerate(docs):
#         all_docs.append({
#             "content": d,
#             "metadata": metas[i] if metas else {},
#             "distances": distances[i] if distances else []
#         })
#     return all_docs