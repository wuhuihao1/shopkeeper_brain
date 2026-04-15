from pymongo import MongoClient

# 连接 MongoDB
client = MongoClient("mongodb://admin:123456@192.168.200.140:27017")

# 选择数据库（不存在则自动创建）
db = client["mydb"]

# 选择集合（不存在则自动创建）
collection = db["students"]

print("连接成功！")

# 插入单条
# result = collection.insert_one({
#     "name": "张三",
#     "age": 20,
#     "major": "计算机科学"
# })
# print(f"插入成功，ID: {result.inserted_id}")

# 插入多条
result = collection.delete_one({"name": "张三"})
print(f"删除 {result.deleted_count} 条")