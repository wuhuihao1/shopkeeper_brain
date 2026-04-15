import os.path
import logging
import shutil
import time
import uuid
from datetime import datetime

from pathlib import Path
from fastapi import UploadFile
from knowledge.core.paths import get_local_base_dir
from knowledge.processor.import_processor.exceptions import FileProcessingError
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.processor.import_processor.main_graph import import_app
from knowledge.utils.task_util import update_task_status, add_running_task, add_done_task, add_node_duration, \
    TASK_STATUS_PROCESSING, \
    TASK_STATUS_COMPLETED, \
    TASK_STATUS_FAILED

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class UpLoadService:
    """
    上传文件服务模块
    """
    def run_import_graph(self, task_id: str, import_file_path: str, file_dir: str, filename: str):
        """
        运行整个图谱函数
        Args:
            task_id: 任务id
            import_file_path: 导入文件的路径
            file_dir: 存放mineru等库解析的文件
        """
        # 更新状态
        update_task_status(task_id, TASK_STATUS_PROCESSING)
        # 定义graph初始状态
        graph_state = {
            "task_id": task_id,
            "import_file_path": import_file_path,
            "file_dir": file_dir
        }
        try:
            for event in import_app.stream(graph_state):
                for key, value in event.items():
                     print(f"当前正在执行的节点--->{key}")
            update_task_status(task_id, TASK_STATUS_COMPLETED)
            self.delete_temp_file(import_file_path, filename)
        except Exception as e:
            logger.error(f"[{task_id}] 执行导入过程中出现异常 原因{str(e)}")
            update_task_status(task_id, TASK_STATUS_FAILED)

    def delete_temp_file(self, import_file_path, filename):
        if import_file_path:
            file_path = Path(import_file_path)
            if file_path.exists():
                file_path.unlink()
                logger.info(f"已删除临时文件: {filename}")
    def get_base_dir(self) -> str:
        """
        定义存放在本地的文件路径
        Returns:
            返回存放文件路径
        """
        return os.path.join(get_local_base_dir(), datetime.now().strftime('%Y%m%d'))

    def process_upload_file(self, file: UploadFile):
        """
        处理文件上传流程
        文件上传本地目录

        Args:
            file: 要上传的文件
        Returns
        file_dir / import_file_path /task_id
        """
        # 生成任务id, 32位uuid
        task_id = str(uuid.uuid4().hex)
        add_running_task(task_id, "upload_file")
        start_time = time.time() #获取上传任务开始时间

        # 2. 生成日期目录并且将日期目录和临时目录拼接到一起,用于存放到该目录
        base_file_dir = self.get_base_dir()

        # 3. 拼接task_id构建文档完整归属目录
        file_dir = os.path.join(base_file_dir, task_id)

        # 4. 保存文件到临时目录
        import_file_path = self.save_upload_file_to_local(file, file_dir)

        # 5. 保存文件到Minio中
        self.save_upload_file_to_minio(import_file_path, file.filename)
        add_done_task(task_id, "upload_file")
        end_time = time.time()
        add_node_duration(task_id, "upload_file", end_time - start_time)

        return file_dir, import_file_path, task_id


    def save_upload_file_to_local(self, file: UploadFile, file_dir: str) -> str:
        """
        将文件临时保存到本地
        Args:
            file: 上传文件的对象
            file_path: 上传文件要保存的位置
        Returns:
            返回一个import_file_path路径
        """
        # 创建文件归属目录
        os.makedirs(file_dir, exist_ok=True)
        # 创建文件存储路径
        import_file_path = os.path.join(get_local_base_dir(), file.filename)
        # 写入文件
        try:
            with open(import_file_path, "wb") as f:
                #将源文件分块复制到目标文件中,防止因为文件过大上传失败
                shutil.copyfileobj(file.file, f)
        except IOError as e:
            logger.error(f'{file.filename}文件写入临时目录失败,原因{str(e)}')
            raise FileProcessingError(message=f'{file.filename}文件写入临时目录失败,原因{str(e)}')

        return import_file_path

    def save_upload_file_to_minio(self, import_file_path: str, filename: str):
        """
        将临时保存在本地的文件上传到minio中
        Args:
            import_file_path: 上传文件的地址
            filename: 上传文件的名称
        """
        # 1. 获取minio客户端
        try:
            minio_client = StorageClients.get_minio_client()
        except ConnectionError as e:
            logger.error(f'minio_client获取失败,原因:{str(e)}')
            return
        # 2. 获取minio相关信息
        bucket_name = os.getenv('MINIO_BUCKET_NAME')
        object_name = f'origin_files/f{datetime.now().strftime("%Y%m%d")}/{filename}'
        # 3. 上传
        try:
            minio_client.fput_object(bucket_name, object_name, import_file_path)
        except Exception as e:
            logger.error(f'上传{filename}文件上传失败,原因:{str(e)}')
