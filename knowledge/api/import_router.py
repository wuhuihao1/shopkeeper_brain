import os.path

import uvicorn
from fastapi import FastAPI, UploadFile, Depends, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from knowledge.core.paths import get_front_page_dir
from knowledge.schema.upload_schema import UploadResponse,TaskStatusResponse
from knowledge.service.upload_service import UpLoadService
from knowledge.core.deps import get_upload_file_service
from knowledge.utils.task_util import get_task_info


# 1. 创建fastapi实例
# 2. 注册路由（将上传请求以及查询导入任务的请求注册到fastapi实例上）
# 3. 利用uvicorn服务器启动fastapi

def create_app():
    app = FastAPI(description='掌柜智库导入应用', version='v1.0')
    #配置跨域
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    front_page_dir = get_front_page_dir()
    app.mount("/front", StaticFiles(directory=front_page_dir))

    register_router(app)

    return app

def register_router(app):
    @app.get('/')
    def index():
        return {"flag": "success"}

    #UploadResponse是一个pydantic库定义的函数,用于类型定义和接口自动接口文档生成
    @app.post('/upload', response_model=UploadResponse)
    def upload_endpoint(file: UploadFile, background_tasks: BackgroundTasks, upload_service: UpLoadService = Depends(get_upload_file_service)):
        """
        上传文件接口
        upload_file,上传的文件
        background_tasks,后台任务,用于异步处理
        upload_service,获取服务实例的方式,UpLoadService文件类型,get_upload_file_service依赖注入的函数,调用它获得实例给到upload_service
        Args:
            upload_file: 文件对象有file和filename属性
            background_tasks: 后台任务
            upload_service: 服务实例

        Returns:

        """
        # 将上传的文件写入到本地临时目录以及远程MinIO
        file_dir, import_file_path, task_id = upload_service.process_upload_file(file)
        # 整个导入的图谱(耗时：节点多【pdf解析很慢】)后台任务慢慢做
        # add_task函数参数: 1. 需要调用的func, 2. func的参数1 3. func的参数2 ...
        background_tasks.add_task(upload_service.run_import_graph, task_id, import_file_path, file_dir, file.filename)
        return UploadResponse(message=f'文件{file.filename}上传成功!', task_id=task_id)

    @app.get('/status/{task_id}')
    def get_task_status_endpoint(task_id: str):
        """
        查询上传任务的状态,每1.5s执行一次
        Args:
            task_id: 查询任务id
        Returns:

        """
        task_info = get_task_info(task_id)
        return TaskStatusResponse(**task_info)

if __name__ == '__main__':
    # param1:fastapi实例
    # param2:启动的服务器地址
    # param3:启动的服务端口
    uvicorn.run(app=create_app(), host="0.0.0.0", port=8000, log_level="info")
