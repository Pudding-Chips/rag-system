import os
import json
import hashlib
import chromadb
from embed import embed_texts
from config import DEFAULT_MODEL

client = chromadb.HttpClient(host='localhost', port=8000)
collection_name = "cdn_qa_v1_768"

def get_content_hash(q, a):
    """结合问题和答案生成唯一 ID，确保内容更新时 ID 随之改变"""
    combined = f"Q:{q}|A:{a}"
    return hashlib.md5(combined.encode('utf-8')).hexdigest()

def fill_database(force_reset=False):
    if force_reset:
        try:
            client.delete_collection(name=collection_name)
            print(f"🗑️ 已清空旧集合: {collection_name}")
        except:
            pass
    
    collection = client.get_or_create_collection(name=collection_name)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(base_dir, "data", "data.json")
    
    if not os.path.exists(json_path):
        print(f"❌ 错误：找不到文件 {json_path}")
        return

    with open(json_path, 'r', encoding='utf-8') as f:
        raw_data = json.load(f)

    all_ids, all_docs, all_metas, all_embs = [], [], [], []
    
    for i, sublist in enumerate(raw_data):
        item = sublist[0] if isinstance(sublist, list) else sublist
        
        q_text = next((v for k, v in item.items() if "question" in k.lower()), "")
        a_text = item.get("answer", "")
        if not q_text: continue

        search_content = f"问题：{q_text} 答案：{a_text}"

        doc_id = get_content_hash(q_text, a_text)

        emb_result = embed_texts(search_content, DEFAULT_MODEL)
        
        vector = emb_result.tolist() if hasattr(emb_result, "tolist") else list(emb_result)
        if isinstance(vector[0], list): vector = vector[0]
            
        if len(vector) != 768:
            print(f"⚠️ 跳过 [{i}]: 维度 {len(vector)} 不匹配预期 768")
            continue

        all_ids.append(doc_id)
        all_docs.append(search_content)
        all_metas.append({
            "answer": a_text, 
            "question": q_text, 
            "update_v": "v2_fulltext" 
        })
        all_embs.append(vector)
        
        print(f"✅ 处理成功: {q_text[:10]}...")

    if all_ids:
        collection.upsert(
            ids=all_ids,
            documents=all_docs,
            metadatas=all_metas,
            embeddings=all_embs
        )
        print(f"🚀 同步完成！库内数据总数: {collection.count()}")

if __name__ == "__main__":
    # 第一次运行新逻辑建议设为 True，清空旧的向量数据
    fill_database(force_reset=True)
