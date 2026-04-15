from functools import cache
from knowledge.service.upload_service import UpLoadService
from knowledge.service.query_service import QueryService

@cache # 缓存注解（将实例对象缓存一份:可能会出现oom:out of memory）实现单例效果,只调用一次其他调用就读缓存
def get_upload_file_service():
    return UpLoadService()

@cache
def get_query_service():
    return QueryService()