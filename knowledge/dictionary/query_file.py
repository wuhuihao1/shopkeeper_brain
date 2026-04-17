from enum import Enum
class IntentType(str, Enum):
    """意图类型枚举"""
    RECOMMEND = "recommend"  # 书籍推荐
    DETAIL = "detail"  # 书籍详情
    SEARCH = "search"  # 内容检索
    QA = "qa"  # 知识问答
    CHAT = "chat"  # 闲聊/问候