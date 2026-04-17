import logging
from typing import Dict, Any, Optional, List, Tuple
from pymilvus import MilvusClient, WeightedRanker, AnnSearchRequest

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# 创建混合检索请求
# ------------------------------------------------------------------
def create_hybrid_search_requests(dense_vector,
                                  sparse_vector,
                                  dense_params=None,
                                  sparse_params=None,
                                  expr=None,
                                  expr_params=None,
                                  limit=5) -> List[AnnSearchRequest]:
    """
    创建混合搜索请求
    Args:
        dense_vector: 稠密向量
        sparse_vector: 稀疏向量
        dense_params: 稠密向量搜索参数，默认为None
        sparse_params: 稀疏向量搜索参数，默认为None
        expr: 查询表达式，默认为None
        expr_params: 查询表达式参数，默认为None
        limit: 返回结果数量限制，默认为5
    Returns:
        包含稠密和稀疏搜索请求的列表
    """
    if dense_vector is None or sparse_vector is None:
        raise ValueError("dense_vector 和 sparse_vector 不能为 None")

    if dense_params is None:
        dense_params = {"metric_type": "COSINE"}
    if sparse_params is None:
        sparse_params = {"metric_type": "IP"}

    dense_req = AnnSearchRequest(
        data=[dense_vector],
        anns_field="dense_vector",
        param=dense_params,
        expr=expr,
        expr_params=expr_params,
        limit=limit
    )

    sparse_req = AnnSearchRequest(
        data=[sparse_vector],
        anns_field="sparse_vector",
        param=sparse_params,
        expr=expr,
        expr_params=expr_params,
        limit=limit
    )

    return [dense_req, sparse_req]


# ------------------------------------------------------------------
# 执行混合检索请求
# ------------------------------------------------------------------
def execute_hybrid_search_query(milvus_client: MilvusClient,
                                collection_name,
                                search_requests,
                                ranker_weights=(0.5, 0.5),
                                norm_score=True,
                                limit=5,
                                output_fields=None,
                                search_params=None):
    """
    执行混合搜索
    Args:
        milvus_client: Milvus客户端
        collection_name: 集合名称
        search_requests: 搜索请求列表
        ranker_weights: 权重排名器的权重，默认为(0.5, 0.5)
        norm_score: 是否对分数进行归一化，默认为True
        limit: 返回结果数量限制，默认为5
        output_fields: 要返回的字段列表，默认为None
        search_params: 搜索参数，默认为None
    Returns:
        搜索结果
    """
    if milvus_client is None:
        raise ValueError("milvus_client 不能为 None")
    if not search_requests:
        raise ValueError("search_requests 不能为 None 或空列表")

    rerank = WeightedRanker(ranker_weights[0], ranker_weights[1], norm_score=norm_score)

    if output_fields is None:
        output_fields = ["book_name", "content", "title", "content_type", "author_name"]

    res = milvus_client.hybrid_search(
        collection_name=collection_name,
        reqs=search_requests,
        ranker=rerank,
        limit=limit,
        output_fields=output_fields,
        search_params=search_params
    )

    total_hits = sum(len(hits) for hits in res) if res else 0
    logger.info(f"Milvus 混合搜索完成，共处理 {len(res) if res else 0} 个查询，总计找到 {total_hits} 个结果")
    return res


# ==================== 过滤表达式构建函数 ====================

def book_names_filter(book_names: List[str]) -> Tuple[str, Dict[str, Any]]:
    """
    构建书名的 Milvus 过滤表达式
    Args:
        book_names: 书名列表，如 ["活着", "三体"]
    Returns:
        (expr, expr_params) 元组
    """
    if not book_names:
        return "", {}

    valid_names = [name for name in book_names if name and name.strip()]
    if not valid_names:
        return "", {}

    if len(valid_names) == 1:
        return f'book_name == "{valid_names[0]}"', {}
    else:
        placeholders = ', '.join([f'"{name}"' for name in valid_names])
        return f'book_name in [{placeholders}]', {}


def content_type_filter(content_types: List[str]) -> Tuple[str, Dict[str, Any]]:
    """
    构建内容类型的 Milvus 过滤表达式
    Args:
        content_types: 内容类型列表，如 ["书籍简介", "作者介绍"]
    Returns:
        (expr, expr_params) 元组
    """
    if not content_types:
        return "", {}

    valid_types = [ct for ct in content_types if ct and ct.strip()]
    if not valid_types:
        return "", {}

    if len(valid_types) == 1:
        return f'content_type == "{valid_types[0]}"', {}
    else:
        placeholders = ', '.join([f'"{ct}"' for ct in valid_types])
        return f'content_type in [{placeholders}]', {}


def author_name_filter(author_name: str) -> Tuple[str, Dict[str, Any]]:
    """
    构建作者名的 Milvus 过滤表达式
    Args:
        author_name: 作者名，如 "余华"
    Returns:
        (expr, expr_params) 元组
    """
    if not author_name or not author_name.strip():
        return "", {}

    return f'author_name == "{author_name.strip()}"', {}


def category_tags_filter(category_tags: List[str]) -> Tuple[str, Dict[str, Any]]:
    """
    构建类别/标签的 Milvus 过滤表达式（VARCHAR 版本）
    使用 LIKE 模糊匹配 JSON 字符串中的标签
    注意：category_tags 存储格式为 JSON 字符串，如 '["科幻", "悬疑"]'
    """
    if not category_tags:
        return "", {}

    valid_tags = [tag for tag in category_tags if tag and tag.strip()]
    if not valid_tags:
        return "", {}

    conditions = []
    for tag in valid_tags:
        # 匹配 JSON 字符串中的标签，如 '["科幻", "悬疑"]' 中包含 "科幻"
        # 需要使用转义的双引号：%\"科幻\"%
        conditions.append(f'category_tags LIKE "%\\"{tag}\\"%"')

    if len(conditions) == 1:
        return conditions[0], {}
    else:
        return f'({" OR ".join(conditions)})', {}

def combine_filters(filters: List[Tuple[str, Dict[str, Any]]]) -> Tuple[str, Dict[str, Any]]:
    """
    组合多个过滤条件（用 AND 连接）
    Args:
        filters: 过滤条件列表，每个元素是 (expr, expr_params) 元组
    Returns:
        (combined_expr, combined_params) 元组
    """
    valid_exprs = []
    combined_params = {}

    for expr, params in filters:
        if expr:
            valid_exprs.append(expr)
            combined_params.update(params)

    if not valid_exprs:
        return "", {}

    if len(valid_exprs) == 1:
        return valid_exprs[0], combined_params
    else:
        return f'({" AND ".join(valid_exprs)})', combined_params