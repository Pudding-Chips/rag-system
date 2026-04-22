import chromadb
client = chromadb.HttpClient(host='localhost', port=8000)
# 看看现在所有的集合名字
print("当前所有集合:", client.list_collections())

# 看看目标集合里有多少条数据
collection = client.get_collection("cdn_qa_v1_768")
print("数据总条数:", collection.count())
# 打印前 2 条看看
print("样例数据:", collection.peek(limit=2))