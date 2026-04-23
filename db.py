import chromadb
from config import DB_HOST, DB_PORT

def get_client():
    return chromadb.HttpClient(host=DB_HOST, port=DB_PORT)

def get_collection(biz, model):
    from config import COLLECTION_TEMPLATE

    name = COLLECTION_TEMPLATE.format(biz=biz, model=model)
    client = get_client()
    return client.get_or_create_collection(name)
