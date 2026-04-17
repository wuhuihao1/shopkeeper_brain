"""查询流程状态类型定义

定义完整的查询状态结构和辅助函数。
"""

from typing import TypedDict, List, Optional, Any
import copy


class QueryGraphState(TypedDict):
    """
    查询流程状态定义（听书平台版本）

    Attributes:
        # 基础信息
        session_id: 会话ID
        task_id: 任务ID
        message_id: 消息ID
        original_query: 原始查询
        is_stream: 是否流式输出

        # 查询处理
        rewritten_query: LLM重写后的查询
        intent: 意图类型 (recommend/detail/search/qa/chat)
        intent_confidence: 意图识别置信度
        intent_reason: 意图判断理由

        # 书名相关
        book_names: 确认的书名列表
        book_name_options: 候选书名列表（供用户选择）

        # 过滤相关
        filter_expr: Milvus过滤表达式
        content_types: 内容类型过滤列表
        category_tags: 类别/标签过滤列表

        # 检索结果
        embedding_chunks: 向量检索结果
        hyde_embedding_chunks: HyDE检索结果
        web_search_docs: 联网搜索结果
        rrf_chunks: RRF融合后的结果
        reranked_docs: 重排序后的文档

        # 答案相关
        prompt: LLM提示词
        answer: 最终答案

        # 历史对话
        history: 历史对话列表

        # 兼容旧字段（逐步废弃）
        item_names: List[str]  # 旧项目商品名，保留兼容
    """
    # 基础信息
    session_id: str
    task_id: str
    message_id: str
    original_query: str
    is_stream: bool

    # 查询处理
    rewritten_query: str
    intent: str
    intent_confidence: float
    intent_reason: str

    # 书名相关
    book_names: List[str]
    book_name_options: List[str]

    # 过滤相关
    filter_expr: str
    content_types: List[str]
    category_tags: List[str]

    # 检索结果
    embedding_chunks: List[Any]
    hyde_embedding_chunks: List[Any]
    web_search_docs: List[Any]
    rrf_chunks: List[Any]
    reranked_docs: List[Any]

    # 答案相关
    prompt: str
    answer: str

    # 历史对话
    history: List[Any]

    # 兼容旧字段
    item_names: List[str]


# ==================== 默认状态 ====================

DEFAULT_STATE: QueryGraphState = {
    # 基础信息
    "session_id": "",
    "task_id": "",
    "message_id": "",
    "original_query": "",
    "is_stream": False,

    # 查询处理
    "rewritten_query": "",
    "intent": "qa",
    "intent_confidence": 0.5,
    "intent_reason": "",

    # 书名相关
    "book_names": [],
    "book_name_options": [],

    # 过滤相关
    "filter_expr": "",
    "content_types": [],
    "category_tags": [],

    # 检索结果
    "embedding_chunks": [],
    "hyde_embedding_chunks": [],
    "web_search_docs": [],
    "rrf_chunks": [],
    "reranked_docs": [],

    # 答案相关
    "prompt": "",
    "answer": "",

    # 历史对话
    "history": [],

    # 兼容旧字段
    "item_names": [],
}


def create_default_state(**overrides) -> QueryGraphState:
    """创建默认状态，支持字段覆盖。

    Args:
        **overrides: 要覆盖的字段键值对。

    Returns:
        新的状态实例，包含默认值和覆盖值。
    """
    state = copy.deepcopy(DEFAULT_STATE)
    state.update(overrides)
    return state


def get_default_state() -> QueryGraphState:
    """获取默认状态副本。

    Returns:
        状态副本，避免修改全局默认值。
    """
    return copy.deepcopy(DEFAULT_STATE)


def update_state(state: QueryGraphState, **updates) -> QueryGraphState:
    """更新状态，返回新状态副本。

    Args:
        state: 原始状态
        **updates: 要更新的字段

    Returns:
        更新后的新状态
    """
    new_state = copy.deepcopy(state)
    new_state.update(updates)
    return new_state


def get_intent_value(state: QueryGraphState) -> str:
    """获取意图值（确保返回字符串）

    Args:
        state: 查询状态

    Returns:
        意图字符串
    """
    intent = state.get("intent", "qa")
    if hasattr(intent, "value"):
        return intent.value
    return str(intent)


def has_books(state: QueryGraphState) -> bool:
    """判断是否有确认的书名

    Args:
        state: 查询状态

    Returns:
        是否有确认的书名
    """
    book_names = state.get("book_names", [])
    return bool(book_names)


def has_answer(state: QueryGraphState) -> bool:
    """判断是否有预置答案

    Args:
        state: 查询状态

    Returns:
        是否有预置答案
    """
    answer = state.get("answer", "")
    return bool(answer)


def has_search_results(state: QueryGraphState) -> bool:
    """判断是否有检索结果

    Args:
        state: 查询状态

    Returns:
        是否有检索结果
    """
    rrf_chunks = state.get("rrf_chunks", [])
    return bool(rrf_chunks)


# 保留全局默认状态引用（向后兼容）
graph_default_state = DEFAULT_STATE