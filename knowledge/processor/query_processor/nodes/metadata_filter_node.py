from typing import List, Dict, Any, Optional, Tuple
from enum import Enum

from knowledge.processor.query_processor.base import BaseNode
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.processor.query_processor.exceptions import StateFieldError


class FilterType(str, Enum):
    """过滤类型枚举"""
    BOOK_NAME = "book_name"  # 按书名过滤
    CONTENT_TYPE = "content_type"  # 按内容类型过滤
    CATEGORY_TAGS = "category_tags"  # 按类别/标签过滤
    AUTHOR_NAME = "author_name"  # 按作者名过滤


# 意图与默认过滤配置的映射
INTENT_FILTER_CONFIG = {
    "recommend": {
        "filter_fields": ["category_tags"],  # 推荐场景主要按标签过滤
        "default_content_types": []  # 不限制内容类型
    },
    "detail": {
        "filter_fields": ["book_name", "content_type"],
        "default_content_types": ["书籍简介", "作者介绍", "有声书信息", "推荐运营资料"]
    },
    "search": {
        "filter_fields": ["book_name", "content_type"],
        "default_content_types": ["听书笔记", "用户评论摘要", "常见问答"]
    },
    "qa": {
        "filter_fields": ["book_name"],  # QA场景可能按书名过滤，不限制内容类型
        "default_content_types": []
    },
    "chat": {
        "filter_fields": [],  # 闲聊不需要过滤
        "default_content_types": []
    }
}


class MetadataFilterNode(BaseNode):
    """
    元数据过滤节点
    根据意图和用户输入，构建 Milvus 过滤表达式
    用于在向量检索前过滤数据
    """
    name = "metadata_filter_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        核心逻辑：构建过滤表达式
        """
        # 1. 获取必要参数
        intent = state.get('intent', 'qa')
        book_names = state.get('book_names', [])  # 从 BookNameConfirmedNode 获取确认的书名
        author_name = state.get('author_name', '')  # 可能从用户输入中提取

        # 2. 获取配置
        filter_config = INTENT_FILTER_CONFIG.get(intent, INTENT_FILTER_CONFIG['qa'])

        # 3. 构建过滤条件列表
        filter_conditions = []

        # 3.1 按书名过滤
        if 'book_name' in filter_config['filter_fields'] and book_names:
            book_name_filter = self._build_book_name_filter(book_names)
            if book_name_filter:
                filter_conditions.append(book_name_filter)

        # 3.2 按内容类型过滤
        if 'content_type' in filter_config['filter_fields']:
            content_types = filter_config.get('default_content_types', [])
            # 如果 state 中有指定 content_type，优先使用
            specified_types = state.get('content_types', [])
            if specified_types:
                content_types = specified_types
            if content_types:
                content_type_filter = self._build_content_type_filter(content_types)
                if content_type_filter:
                    filter_conditions.append(content_type_filter)

        # 3.3 按类别/标签过滤（推荐场景）
        if 'category_tags' in filter_config['filter_fields']:
            category_tags = state.get('category_tags', []) or self._extract_category_from_query(state)
            if category_tags:
                category_filter = self._build_category_filter(category_tags)
                if category_filter:
                    filter_conditions.append(category_filter)

        # 3.4 按作者名过滤
        if 'author_name' in filter_config['filter_fields'] and author_name:
            author_filter = self._build_author_filter(author_name)
            if author_filter:
                filter_conditions.append(author_filter)

        # 4. 组合过滤表达式
        filter_expr = self._combine_filters(filter_conditions)

        # 5. 保存到 state
        state['filter_expr'] = filter_expr
        state['filter_conditions'] = filter_conditions

        self.logger.info(
            f"过滤条件构建完成 - 意图: {intent}, "
            f"书名单: {book_names}, "
            f"过滤表达式: {filter_expr}"
        )

        return state

    def _build_book_name_filter(self, book_names: List[str]) -> Optional[str]:
        """
        构建书名过滤表达式
        Args:
            book_names: 书名列表
        Returns:
            Milvus 过滤表达式，如: 'book_name in ["活着", "三体"]'
        """
        if not book_names:
            return None

        # 过滤掉空字符串
        valid_names = [name for name in book_names if name and name.strip()]
        if not valid_names:
            return None

        # 转义书名中的特殊字符（引号）
        escaped_names = [name.replace('"', '\\"') for name in valid_names]

        if len(escaped_names) == 1:
            return f'book_name == "{escaped_names[0]}"'
        else:
            names_str = ', '.join([f'"{name}"' for name in escaped_names])
            return f'book_name in [{names_str}]'

    def _build_content_type_filter(self, content_types: List[str]) -> Optional[str]:
        """
        构建内容类型过滤表达式
        Args:
            content_types: 内容类型列表，如 ["书籍简介", "作者介绍"]
        Returns:
            Milvus 过滤表达式，如: 'content_type in ["书籍简介", "作者介绍"]'
        """
        if not content_types:
            return None

        valid_types = [ct for ct in content_types if ct and ct.strip()]
        if not valid_types:
            return None

        escaped_types = [ct.replace('"', '\\"') for ct in valid_types]

        if len(escaped_types) == 1:
            return f'content_type == "{escaped_types[0]}"'
        else:
            types_str = ', '.join([f'"{ct}"' for ct in escaped_types])
            return f'content_type in [{types_str}]'

    def _build_category_filter(self, category_tags: List[str]) -> Optional[str]:
        """构建类别/标签过滤表达式（VARCHAR 版本）"""
        if not category_tags:
            return None

        valid_tags = [tag for tag in category_tags if tag and tag.strip()]
        if not valid_tags:
            return None

        conditions = []
        for tag in valid_tags:
            # 使用 LIKE 匹配，因为 category_tags 是 VARCHAR 类型
            conditions.append(f'category_tags LIKE "%\\"{tag}\\"%"')

        if len(conditions) == 1:
            return conditions[0]
        else:
            return f'({" OR ".join(conditions)})'

    def _build_author_filter(self, author_name: str) -> Optional[str]:
        """
        构建作者名过滤表达式
        Args:
            author_name: 作者名
        Returns:
            Milvus 过滤表达式，如: 'author_name == "余华"'
        """
        if not author_name or not author_name.strip():
            return None

        escaped_name = author_name.replace('"', '\\"')
        return f'author_name == "{escaped_name}"'

    def _combine_filters(self, conditions: List[str]) -> str:
        """
        组合多个过滤条件
        Args:
            conditions: 过滤条件列表
        Returns:
            组合后的过滤表达式，如: '(book_name == "活着") AND (content_type in ["书籍简介"])'
        """
        if not conditions:
            return ""

        if len(conditions) == 1:
            return conditions[0]

        # 多个条件用 AND 连接
        return f'({" AND ".join(conditions)})'

    def _extract_category_from_query(self, state: QueryGraphState) -> List[str]:
        """
        从用户查询中提取类别/标签（推荐场景）
        这是一个简化版本，实际可以通过 LLM 提取或使用关键词匹配
        Args:
            state: 查询状态
        Returns:
            提取的标签列表
        """
        query = state.get('rewritten_query', '') or state.get('original_query', '')
        if not query:
            return []

        # 预定义的类别关键词映射
        category_keywords = {
            "科幻": ["科幻", "science fiction", "sci-fi"],
            "悬疑": ["悬疑", "推理", "侦探"],
            "言情": ["言情", "爱情", "恋爱"],
            "武侠": ["武侠", "江湖"],
            "历史": ["历史", "古代"],
            "奇幻": ["奇幻", "魔幻", "魔法"],
            "恐怖": ["恐怖", "惊悚"],
            "儿童": ["儿童", "童话", "亲子"],
            "教育": ["教育", "学习", "知识"],
            "文学": ["文学", "经典", "名著"],
            "传记": ["传记", "人物"],
            "励志": ["励志", "成长", "成功"],
        }

        # 场景关键词映射
        scene_keywords = {
            "通勤": ["通勤", "路上", "地铁", "公交"],
            "睡前": ["睡前", "助眠", "催眠"],
            "运动": ["运动", "跑步", "健身"],
            "做家务": ["家务", "做饭", "打扫"],
        }

        extracted_tags = []
        query_lower = query.lower()

        # 匹配类别
        for tag, keywords in category_keywords.items():
            for kw in keywords:
                if kw in query_lower:
                    extracted_tags.append(tag)
                    break

        # 匹配场景
        for tag, keywords in scene_keywords.items():
            for kw in keywords:
                if kw in query_lower:
                    extracted_tags.append(tag)
                    break

        # 去重
        return list(set(extracted_tags))


class MetadataFilterHelper:
    """
    过滤表达式辅助类
    提供常用的过滤表达式构建方法，供其他节点使用
    """

    @staticmethod
    def build_expr_for_detail(book_name: str) -> str:
        """构建详情场景的过滤表达式"""
        escaped_name = book_name.replace('"', '\\"')
        content_types = ["书籍简介", "作者介绍", "有声书信息", "推荐运营资料"]
        types_str = ', '.join([f'"{ct}"' for ct in content_types])
        return f'(book_name == "{escaped_name}") AND (content_type in [{types_str}])'

    @staticmethod
    def build_expr_for_search(book_name: Optional[str] = None) -> str:
        """构建检索场景的过滤表达式"""
        content_types = ["听书笔记", "用户评论摘要", "常见问答"]
        types_str = ', '.join([f'"{ct}"' for ct in content_types])

        if book_name:
            escaped_name = book_name.replace('"', '\\"')
            return f'(book_name == "{escaped_name}") AND (content_type in [{types_str}])'
        else:
            return f'content_type in [{types_str}]'

    @staticmethod
    def build_expr_for_recommend(category_tags: List[str]) -> str:
        """构建推荐场景的过滤表达式"""
        if not category_tags:
            return ""

        conditions = []
        for tag in category_tags:
            escaped_tag = tag.replace('"', '\\"')
            conditions.append(f'JSON_CONTAINS(category_tags, "{escaped_tag}")')

        if len(conditions) == 1:
            return conditions[0]
        else:
            return f'({" OR ".join(conditions)})'

    @staticmethod
    def build_expr_for_book_name(book_names: List[str]) -> str:
        """构建按书名过滤的表达式"""
        if not book_names:
            return ""

        escaped_names = [name.replace('"', '\\"') for name in book_names]

        if len(escaped_names) == 1:
            return f'book_name == "{escaped_names[0]}"'
        else:
            names_str = ', '.join([f'"{name}"' for name in escaped_names])
            return f'book_name in [{names_str}]'


if __name__ == '__main__':
    # 测试代码
    class MockConfig:
        pass


    node = MetadataFilterNode()
    node.config = MockConfig()

    test_cases = [
        {
            "name": "推荐场景 - 科幻",
            "state": {
                "intent": "recommend",
                "original_query": "推荐几本好看的科幻小说",
                "book_names": []
            }
        },
        {
            "name": "详情场景 - 活着",
            "state": {
                "intent": "detail",
                "book_names": ["活着"],
                "original_query": "《活着》讲什么"
            }
        },
        {
            "name": "检索场景 - 红楼梦笔记",
            "state": {
                "intent": "search",
                "book_names": ["红楼梦"],
                "original_query": "查询红楼梦的听书笔记"
            }
        },
        {
            "name": "问答场景 - 无书名",
            "state": {
                "intent": "qa",
                "book_names": [],
                "original_query": "什么是生命韧性"
            }
        },
        {
            "name": "推荐场景 - 通勤",
            "state": {
                "intent": "recommend",
                "original_query": "通勤适合听什么书",
                "book_names": []
            }
        }
    ]

    print("=" * 60)
    print("元数据过滤节点测试")
    print("=" * 60)

    for test in test_cases:
        result = node.process(test["state"])
        print(f"\n【{test['name']}】")
        print(f"  意图: {result.get('intent')}")
        print(f"  书名单: {result.get('book_names')}")
        print(f"  过滤表达式: {result.get('filter_expr')}")
        print("-" * 40)