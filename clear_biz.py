import chromadb
from config import DEFAULT_BIZ  # 确保这里和 fill_db.py 引用一致

# 1. 初始化客户端
client = chromadb.HttpClient(host='localhost', port=8000)

# 2. 这里的名字一定要和你之前灌库时的一模一样
target_collection_name = 'cdn_text2vec_v1' 

# 3. 执行删除逻辑
try:
    # 获取当前所有集合，看看我们要删的在不在
    existing_collections = [c.name for c in client.list_collections()]
    
    if target_collection_name in existing_collections:
        print(f"🔍 找到集合 '{target_collection_name}'，正在清理...")
        client.delete_collection(name=target_collection_name)
        print(f"✅ 清理完成！集合 '{target_collection_name}' 已从数据库移除。")
    else:
        print(f"⚠️ 数据库中没找到名为 '{target_collection_name}' 的集合。")
        print(f"当前存在的集合有: {existing_collections}")

except Exception as e:
    print(f"❌ 发生错误: {e}")