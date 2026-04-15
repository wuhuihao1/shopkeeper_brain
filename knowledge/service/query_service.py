import uuid, logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from typing import List,Dict,Any
from knowledge.processor.query_processor.main_graph import query_app
from knowledge.utils.task_util import update_task_status, TASK_STATUS_PROCESSING, TASK_STATUS_FAILED, \
    TASK_STATUS_COMPLETED
from knowledge.utils.task_util import get_task_result
from knowledge.utils.mongo_history_util import get_recent_messages
from knowledge.utils.mongo_history_util import clear_history
from knowledge.utils.mongo_history_util import list_sessions as db_list_sessions
from knowledge.utils.mongo_history_util import delete_session as db_delete_session


class QueryService:
    @staticmethod
    def generate_session_id():
        """生成session_id"""
        return str(uuid.uuid4())
    @staticmethod
    def generate_task_id():
        """生成task_id"""
        return str(uuid.uuid4().hex[:12])

    def run_query_graph(self, session_id: str, task_id: str, query: str, is_stream: bool):
        """
        运行查询流程的pineline
        Args:
            session_id:  会话id
            task_id:     任务id
            query:       查询问题
            is_stream:   是否是流式
        """
        #更新节点流程
        update_task_status(task_id=task_id, status_name=TASK_STATUS_PROCESSING)
        #节点初始化
        query_init_state = {
            "session_id": session_id,
            "task_id": task_id,
            "original_query": query,
            "is_stream": is_stream
        }
        try:
            query_app.invoke(query_init_state)
            update_task_status(task_id=task_id, status_name=TASK_STATUS_COMPLETED)
        except Exception as e:
            logger.error(f"运行查询流程出现异常:{str(e)}")
            update_task_status(task_id=task_id, status_name=TASK_STATUS_FAILED)

    def get_task_result(self, task_id: str):
        """获取节点结果"""
        answer = get_task_result(task_id=task_id, key="answer")
        return answer

    def get_history(self, session_id: str, limit: int = 50) -> List[Dict[str, Any]]:
        records = get_recent_messages(session_id=session_id, limit=limit)
        return [
            {
                "_id": str(r.get("_id", "")),
                "session_id": r.get("session_id"),
                "role": r.get("role"),
                "text": r.get("text"),
                "rewritten_query": r.get("rewritten_query"),
                "item_names": r.get("item_names", []),
                "ts": r.get("ts"),
            }
            for r in records
        ]
    def clear_history(self, session_id: str) -> int:
        return clear_history(session_id)

    def list_sessions(self) -> List[Dict[str, Any]]:
        return db_list_sessions()

    def delete_session(self, session_id: str) -> int:
        return db_delete_session(session_id)
