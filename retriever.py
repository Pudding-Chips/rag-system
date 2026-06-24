"""
接收用户的 query,
转化为向量后去 ChromaDB 中做初筛检索，默认快速捞出前 5 条最相关的候选文档。
"""
from db import get_client
from embed import embed_texts
from config import COLLECTION_TEMPLATE, DEFAULT_BIZ

class SimpleDocument:
    def __init__(self, content, metadata):
        self.page_content = content
        self.metadata = metadata

def retrieve(query, workspace_id, model="text2vec", biz=DEFAULT_BIZ, n_results=10):
    if not workspace_id:
        print("error: workspace_id is empty")
        return []
    
    client = get_client()
    
    target_collection_name = COLLECTION_TEMPLATE.format(biz=biz)
    
    try:
        collection = client.get_collection(name=target_collection_name)
    except Exception as e:
        print(f"error: Unable to get collection {target_collection_name}, please check if fill.py running correctly. error: {e}")
        return []
    
    query_list = embed_texts(query, model)
    
    if len(query_list) > 0 and not isinstance(query_list[0], list):
        final_emb = [query_list]
    else:
        final_emb = query_list

    target_id = str(workspace_id).strip()
    print(f"[RAG QUERY] workspace={target_id} | biz={biz} | query={query[:40]}...")

    #filter (对齐fill.py)
    where_filter = {"workspace_id": {"$eq": target_id}}

    try:
        results = collection.query(
            query_embeddings=final_emb,
            n_results=n_results,
            where=where_filter,
            include=["documents", "metadatas", "distances"]
        )
    except Exception as e:
        print(f"Error: Chromadb fail to retrieve: {e}")
        return []
    
    docs = results.get("documents", [[]])[0] if results.get("documents") else []
    metas = results.get("metadatas", [[]])[0] if results.get("metadatas") else []
    dists = results.get("distances", [[]])[0] if results.get("distances") else []
    
    print(f"Target collection: {target_collection_name} | Quantity: {len(docs)}")

    all_docs = []
    for i in range(len(docs)):
        meta = metas[i] if i < len(metas) and metas[i] else {}
        raw_dist = dists[i] if i < len(dists) else 1.0

        meta["relevance_score"] = max(0, 1 - raw_dist)

        all_docs.append(SimpleDocument(
            content=docs[i],
            metadata=meta
        ))        
    return all_docs

if __name__ == "__main__":
    test_docs = retrieve(
        query="怎么修改密码？", 
        workspace_id="my_prod_workspace_01", 
        n_results=3
    )
    
    print(f"\Found {len(test_docs)} result")
    for doc in test_docs:
        print(f"Content: {doc.page_content}")
        print(f"Similarity: {doc.metadata.get('relevance_score'):.4f}")
        print("-" * 30)
