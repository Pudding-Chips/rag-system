import json
import chromadb
from db import get_client
from embed import embed_texts

class SimpleDocument:
    def __init__(self, content, metadata):
        self.page_content = content
        self.metadata = metadata

def retrieve(query, model="text2vec", biz="cdn", n_results=5):
    client = get_client()
    
    target_collection_name = "cdn_qa_v1_768" 
    
    try:
        collection = client.get_collection(name=target_collection_name)
    except Exception as e:
        print(f"[ERROR] 无法获取集合 {target_collection_name}，请检查 fill.py 是否运行成功。错误: {e}")
        return []
    
    query_embs = embed_texts(query, model)
    
    if hasattr(query_embs, "tolist"):
        query_list = query_embs.tolist()
    else:
        query_list = list(query_embs)

    if len(query_list) > 0 and not isinstance(query_list[0], list):
        final_emb = [query_list]
    else:
        final_emb = query_list

    results = collection.query(
        query_embeddings=final_emb,
        n_results=n_results,
        include=["documents", "metadatas"]
    )
    
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    
    print(f"[DEBUG] 目标集合: {target_collection_name} | 召回数量: {len(docs)}")

    all_docs = []
    for i in range(len(docs)):
        doc_obj = SimpleDocument(
            content=docs[i],
            metadata=metas[i] if metas else {}
        )
        all_docs.append(doc_obj)
        
    return all_docs
