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

def get_content_hash(q, a, chat_id):
    """结合问题和答案生成唯一 ID, 确保内容更新时 ID 随之改变"""
    combined = f"WS:{chat_id}|Q:{q}|A:{a}"
    return hashlib.md5(combined.encode('utf-8')).hexdigest()

def fill_database(chat_id, biz=DEFAULT_BIZ, json_file="data/data.json", force_reset=False, agent_id="unknown", group_id=""):
    batch_size = 100

    if not chat_id:
        print("error: chat_id invalid")
        return {"status": "error", "message": "Missing chat_id"} 

    current_collection_name = COLLECTION_TEMPLATE.format(biz=biz)

    if force_reset:
        try:
            client.delete_collection(name=current_collection_name)
            print(f"Old collection cleared: {current_collection_name}")
        except Exception as e:
            print(f"error: {e}")

    collection = client.get_or_create_collection(name=current_collection_name)

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

    all_ids, all_docs, all_metas, all_embs, all_raw_contents = [], [], [], [], []

    for i, sublist in enumerate(raw_data):
        item = sublist[0] if isinstance(sublist, list) else sublist
        if not isinstance(item, dict): 
            continue

        raw_question = next((v for k, v in item.items() if "question" in k.lower()), None)
            
        # 2. 获取并转换标准答案为字符串
        raw_answer = item.get("answer", "")
        if isinstance(raw_answer, dict):
            a_text = json.dumps(raw_answer, ensure_ascii=False, indent=2)
        else:
            a_text = str(raw_answer).strip()

        # 3. 【重点检查这里】这一行必须比上面的 for 语句多缩进 4 个空格（处于循环内部）
        if not raw_question or not a_text: 
            continue

        questions = raw_question if isinstance(raw_question, list) else[raw_question]

        # 4. 遍历同义句逻辑（同样需要正确缩进）
        for q_text in questions:
            q_text = str(q_text).strip()
            if not q_text: 
                continue

            search_content = q_text
            
            combined_str = f"Question: {q_text} Answer: {a_text}"
            doc_id = get_content_hash(q_text, a_text, chat_id)

            all_ids.append(doc_id)
            all_docs.append(search_content)
            all_raw_contents.append(search_content)
            all_metas.append({
                "chat_id": chat_id,
                "biz": biz,
                "agent_id": agent_id,
                "group_id": str(group_id),
                "answer": a_text, 
                "question": q_text, 
                "update_v": "v2_metadata_isolation",
                "updated_at": datetime.now().isoformat()
            })

    total_added = len(all_ids)
    if total_added > 0:
        try:
            print(f"Generating vectors for {total_added} texts.")
            emb_result = embed_texts(all_raw_contents, DEFAULT_MODEL)

            if hasattr(emb_result, "tolist"):
                all_embs = emb_result.tolist()
            else:
                import numpy as np
                all_embs = np.array(emb_result).tolist()

            if len(all_embs) > 0 and len(all_embs[0]) != 768:
                print(f"error: The model output has {len(all_embs[0])}, which does not match the expected 768 dimensions!")
                return {"status": "error", "message": "Embedding dimension mismatch"}

            

        except Exception as e:
            print(f"error: Fail to embed batches: {e}")
            return {"status": "error", "message": "Embedding failed"}

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
        chat_id="-1001234567890", 
        force_reset=True
    )
    print(f"End with: {result}")