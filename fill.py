import os
import json
import hashlib
import chromadb
from embed import embed_texts
from config import DEFAULT_MODEL

# 1. 初始化客户端
client = chromadb.HttpClient(host='localhost', port=8000)
collection_name = "cdn_qa_v1_768"

def get_content_hash(q, a):
    """结合问题和答案生成唯一 ID，确保内容更新时 ID 随之改变"""
    combined = f"Q:{q}|A:{a}"
    return hashlib.md5(combined.encode('utf-8')).hexdigest()

def fill_database(force_reset=False):
    # 如果需要彻底重置（建议在逻辑大改后执行一次 True）
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
        
        # 1. 提取 Q 和 A
        q_text = next((v for k, v in item.items() if "question" in k.lower()), "")
        a_text = item.get("answer", "")
        if not q_text: continue

        # 2. 【关键修改】构造用于检索的混合文本
        # 将问题和答案拼接，让向量模型同时学习两者的特征
        search_content = f"问题：{q_text} 答案：{a_text}"

        # 3. 生成 ID（基于 Q+A）
        doc_id = get_content_hash(q_text, a_text)

        # 4. 【关键修改】用混合文本生成向量
        emb_result = embed_texts(search_content, DEFAULT_MODEL)
        
        # 向量格式处理
        vector = emb_result.tolist() if hasattr(emb_result, "tolist") else list(emb_result)
        if isinstance(vector[0], list): vector = vector[0]

        # 5. 维度检查（根据你的模型应为 768）
        if len(vector) != 768:
            print(f"⚠️ 跳过 [{i}]: 维度 {len(vector)} 不匹配预期 768")
            continue

        all_ids.append(doc_id)
        all_docs.append(search_content) # 存入混合文本
        all_metas.append({
            "answer": a_text, 
            "question": q_text, # 把原始问题也存入 metadata 方便 api 调用
            "update_v": "v2_fulltext" 
        })
        all_embs.append(vector)
        
        print(f"✅ 处理成功: {q_text[:10]}...")

    # 6. 批量写入
    if all_ids:
        collection.upsert(
            ids=all_ids,
            documents=all_docs,
            metadatas=all_metas,
            embeddings=all_embs
        )
        print(f"🚀 同步完成！库内数据总数: {collection.count()}")

if __name__ == "__main__":
    # 第一次运行新逻辑建议设为 True，清空旧的“仅问题”向量数据
    fill_database(force_reset=True)



# import os
# import json
# import hashlib
# import chromadb
# from embed import embed_texts
# from config import DEFAULT_MODEL

# # 1. 初始化客户端
# client = chromadb.HttpClient(host='localhost', port=8000)

# # 2. 集合配置
# collection_name = "cdn_qa_v1_768"

# def get_content_hash(text):
#     return hashlib.md5(text.encode('utf-8')).hexdigest()

# def fill_database(force_reset=False):
#     # 如果 force_reset 为 True，先删掉整个集合再重建，确保没有任何脏数据
#     if force_reset:
#         try:
#             client.delete_collection(name=collection_name)
#             print(f"🗑️ 已清空旧集合: {collection_name}")
#         except:
#             pass
    
#     collection = client.get_or_create_collection(name=collection_name)

#     base_dir = os.path.dirname(os.path.abspath(__file__))
#     json_path = os.path.join(base_dir, "data", "data.json")
    
#     with open(json_path, 'r', encoding='utf-8') as f:
#         raw_data = json.load(f)

#     all_ids, all_docs, all_metas, all_embs = [], [], [], []
    
#     for i, sublist in enumerate(raw_data):
#         item = sublist[0] if isinstance(sublist, list) else sublist
#         q_text = next((v for k, v in item.items() if "question" in k.lower()), "")
#         a_text = item.get("answer", "")

#         if not q_text: continue

#         # 生成 ID
#         doc_id = get_content_hash(q_text)
        
#         # 生成向量
#         emb_result = embed_texts(q_text, DEFAULT_MODEL)
#         vector = emb_result.tolist() if hasattr(emb_result, "tolist") else list(emb_result)
#         if isinstance(vector[0], list): vector = vector[0]

#         if len(vector) != 768: continue

#         all_ids.append(doc_id)
#         all_docs.append(q_text)
#         # 加入时间戳，确保 metadata 发生物理变化，强制数据库触发更新
#         all_metas.append({"answer": a_text, "version": "2026.04.22.v1"})
#         all_embs.append(vector)
        
#     if all_ids:
#         # 使用 upsert 执行覆盖逻辑
#         collection.upsert(
#             ids=all_ids,
#             documents=all_docs,
#             metadatas=all_metas,
#             embeddings=all_embs
#         )
#         print(f"🚀 同步成功！当前集合内共有 {collection.count()} 条唯一数据。")

# if __name__ == "__main__":
#     # 💡 以后如果你想彻底推倒重来，就把这里改成 True
#     # 如果只是日常小改，用 False 即可
#     fill_database(force_reset=True)