import logging
from typing import List, Dict, Any, Optional
from datetime import datetime
from bson import ObjectId
from pymongo.collection import Collection
from pymongo import DESCENDING

from knowledge.utils.client.storage_clients import StorageClients

logger = logging.getLogger(__name__)


def _get_collection() -> Collection:
    """获取 chat_message 集合"""
    return StorageClients.get_mongo_db()["chat_message"]


def save_chat_message(
        session_id: str,
        role: str,
        text: str,
        rewritten_query: str = "",
        book_names: Optional[List[str]] = None,
        intent: str = "",
        message_id: str = None,
) -> str:
    """
    保存或更新聊天消息到MongoDB
    Args:
        session_id: 会话ID
        role: 角色（user/assistant）
        text: 消息内容
        rewritten_query: 重写后的查询
        book_names: 书名列表（新项目）
        intent: 意图类型（recommend/detail/search/qa/chat）
        message_id: 消息ID（如果提供则更新，否则新增）
    Returns:
        消息ID
    """
    ts = datetime.now().timestamp()

    document = {
        "session_id": session_id,
        "role": role,
        "text": text,
        "rewritten_query": rewritten_query,
        "book_names": book_names or [],  # 改为 book_names
        "intent": intent,                # 新增意图字段
        "ts": ts,
    }

    collection = _get_collection()
    if message_id:
        collection.update_one(
            {"_id": ObjectId(message_id)},
            {"$set": document},
        )
        return message_id
    else:
        result = collection.insert_one(document)
        return str(result.inserted_id)


def get_recent_messages(session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    获取最近的历史消息
    Args:
        session_id: 会话ID
        limit: 返回数量限制
    Returns:
        消息列表（按时间倒序）
    """
    try:
        cursor = (
            _get_collection()
            .find({"session_id": session_id})
            .sort("ts", DESCENDING)
            .limit(limit)
        )
        return list(cursor)
    except Exception as e:
        logger.error(f"获取历史消息失败: {e}")
        return []


def get_recent_messages_ordered(session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
    """
    获取最近的历史消息（按时间正序，便于构建对话上下文）
    Args:
        session_id: 会话ID
        limit: 返回数量限制
    Returns:
        消息列表（按时间正序）
    """
    try:
        cursor = (
            _get_collection()
            .find({"session_id": session_id})
            .sort("ts", DESCENDING)
            .limit(limit)
        )
        messages = list(cursor)
        # 反转为正序
        messages.reverse()
        return messages
    except Exception as e:
        logger.error(f"获取历史消息失败: {e}")
        return []


def list_sessions() -> List[Dict[str, Any]]:
    """
    列出所有会话（按最后活跃时间倒序）
    Returns:
        会话列表，包含 session_id, last_text, last_ts, count
    """
    pipeline = [
        {"$sort": {"ts": -1}},
        {"$group": {
            "_id": "$session_id",
            "last_text": {"$first": "$text"},
            "last_ts": {"$first": "$ts"},
            "count": {"$sum": 1}
        }},
        {"$sort": {"last_ts": -1}}
    ]
    try:
        results = list(_get_collection().aggregate(pipeline))
        return [
            {
                "session_id": r["_id"],
                "last_text": r.get("last_text", ""),
                "last_ts": r.get("last_ts"),
                "count": r.get("count", 0),
            }
            for r in results
        ]
    except Exception as e:
        logger.error(f"列出会话失败: {e}")
        return []


def delete_session(session_id: str) -> int:
    """删除指定会话的所有历史记录"""
    return clear_history(session_id)


def clear_history(session_id: str) -> int:
    """
    清除指定会话的所有历史记录
    Args:
        session_id: 会话ID
    Returns:
        删除的记录数
    """
    try:
        result = _get_collection().delete_many({"session_id": session_id})
        logger.info(f"已删除会话 {session_id} 的 {result.deleted_count} 条消息")
        return result.deleted_count
    except Exception as e:
        logger.error(f"清除会话历史失败 {session_id}: {e}")
        return 0


def get_session_messages(session_id: str, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
    """
    分页获取会话消息
    Args:
        session_id: 会话ID
        limit: 返回数量
        offset: 偏移量
    Returns:
        消息列表（按时间正序）
    """
    try:
        cursor = (
            _get_collection()
            .find({"session_id": session_id})
            .sort("ts", 1)
            .skip(offset)
            .limit(limit)
        )
        return list(cursor)
    except Exception as e:
        logger.error(f"获取会话消息失败: {e}")
        return []