"""
负责初始化并向上层组件提供 ChromaDB 的 HttpClient
以及具体的Collection对象。
"""
import chromadb
from config import DB_HOST, DB_PORT, COLLECTION_TEMPLATE

_client_instance = None
_collection_cache = {}

def get_client():
    global _client_instance
    if _client_instance is None:
        print(f"Initializing http://{DB_HOST}:{DB_PORT}")
        _client_instance = chromadb.HttpClient(host=DB_HOST, port=DB_PORT)
    return _client_instance

def get_collection(biz):
    global _collection_cache

    name = COLLECTION_TEMPLATE.format(biz=biz)

    if name not in _collection_cache:
        client = get_client()
        print(f"Connecting/Creating ChromaDB: {name}")
        _collection_cache[name] = client.get_or_create_collection(name)

    return _collection_cache[name]
