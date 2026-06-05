"""
解析本地 data.json,
调用 embed.py 生成向量，并批量写入 ChromaDB。
"""
import os
import json
import hashlib
import chromadb
from datetime import datetime
from embed import embed_texts
from config import DEFAULT_MODEL, COLLECTION_TEMPLATE, DEFAULT_BIZ

client = chromadb.HttpClient(host='localhost', port=8000)
target_collection_name = COLLECTION_TEMPLATE.format(biz=DEFAULT_BIZ)

def get_content_hash(q, a, workspace_id):
    """结合问题和答案生成唯一 ID, 确保内容更新时 ID 随之改变"""
    combined = f"WS:{workspace_id}|Q:{q}|A:{a}"
    return hashlib.md5(combined.encode('utf-8')).hexdigest()

def fill_database(workspace_id, biz=DEFAULT_BIZ, json_file="data/data.json", force_reset=False, agent_id="unknown", group_id=""):
    batch_size = 100
    
    if not workspace_id:
        print("error: workspace_id invalid")
        return {"status": "error", "message": "Missing workspace_id"}
    
    if force_reset:
        try:
            client.delete_collection(name=target_collection_name)
            print(f"Old collection cleared: {target_collection_name}")
        except Exception as e:
            print(f"error: {e}")
    
    collection = client.get_or_create_collection(name=target_collection_name)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(base_dir, json_file)
    
    if not os.path.exists(json_path):
        print(f"error: Can't found {json_path}")
        return {"status": "error", "message": f"File Not Found: {json_path}"}

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
    except json.JSONDecodeError:
        print("error: JSON file broke")
        return {"status": "error", "message": "Invalid JSON format"}

    all_ids, all_docs, all_metas, all_embs = [], [], [], []
    
    for i, sublist in enumerate(raw_data):
        item = sublist[0] if isinstance(sublist, list) else sublist
        if not isinstance(item, dict): continue

        q_text = next((v for k, v in item.items() if "question" in k.lower()), "").strip()
        a_text = item.get("answer", "").strip()
        if not q_text: continue

        search_content = f"Question: {q_text} Answer: {a_text}"
        doc_id = get_content_hash(q_text, a_text, workspace_id)

        try:
            emb_result = embed_texts(search_content, DEFAULT_MODEL)
            vector = emb_result.tolist() if hasattr(emb_result, "tolist") else list(emb_result)
            if isinstance(vector[0], list): vector = vector[0]
                
            if len(vector) != 768:
                print(f"Skip [{i}]: dimensions {len(vector)} does not match expectations 768")
                continue
        except Exception as e:
            print(f"error: Fail to embed: {e}")
            continue

        all_ids.append(doc_id)
        all_docs.append(search_content)
        all_metas.append({
            "workspace_id": workspace_id,
            "biz": biz,
            "agent_id": agent_id,
            "group_id": str(group_id),
            "answer": a_text, 
            "question": q_text, 
            "update_v": "v2_metadata_isolation",
            "updated_at": datetime.now().isoformat()
        })
        all_embs.append(vector)
        
    total_added = len(all_ids)
    if total_added > 0:
        print(f"Start writing in batches, total: {total_added}...")
            
        for step in range(0, total_added, batch_size):
            end_idx = step + batch_size
            collection.upsert(
                ids=all_ids[step:end_idx],
                documents=all_docs[step:end_idx],
                metadatas=all_metas[step:end_idx],
                embeddings=all_embs[step:end_idx]
            )
            print(f"Added: {min(end_idx, total_added)}/{total_added}")
                
        return {"status": "success", "count": collection.count(), "added": total_added}
            
    return {"status": "no-data"}

if __name__ == "__main__":
    result = fill_database(
        workspace_id="openclaw_group_A", 
        force_reset=True
    )
    print(f"End with: {result}")
