from typing import Tuple, List, Dict, Any, Union

from knowledge.processor.query_processor.base import BaseNode
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.processor.query_processor.exceptions import StateFieldError
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
from knowledge.utils.milvus_util import create_hybrid_search_requests, execute_hybrid_search_query, book_names_filter


class HybridVectorSearchNode(BaseNode):
    """
    混合向量检索节点（听书平台版本）
    1. 对用户查询进行向量化
    2. 在Milvus中进行混合检索（稠密+稀疏）
    3. 支持元数据过滤
    """
    name = 'hybrid_vector_search_node'

    def process(self, state: QueryGraphState) -> Union[QueryGraphState, Dict[str, Any]]:
        # 1. 参数校验
        rewritten_query, book_names = self._validate_state(state)

        # 2. 获取过滤表达式（从 MetadataFilterNode 构建）
        filter_expr = state.get('filter_expr', '')

        # 3. 获取嵌入模型客户端
        try:
            bge_client = AIClients.get_bge_m3_client()
        except ConnectionError as e:
            self.logger.error(f'获取嵌入模型客户端失败, 原因{str(e)}')
            return {"embedding_chunks": []}

        # 4. 获取Milvus客户端
        try:
            milvus_client = StorageClients.get_milvus_client()
        except ConnectionError as e:
            self.logger.error(f'获取milvus客户端失败, 原因{str(e)}')
            return {"embedding_chunks": []}

        # 5. 对查询进行向量化
        try:
            embed_vector = generate_bge_m3_hybrid_vectors(
                model=bge_client,
                embedding_documents=[rewritten_query]
            )
        except Exception as e:
            self.logger.error(f'用户问题{rewritten_query}嵌入向量获取失败, 原因{str(e)}')
            return {"embedding_chunks": []}

        # 6. 构建书名过滤表达式
        expr, expr_params = book_names_filter(book_names)

        # 7. 合并过滤表达式
        if filter_expr and expr:
            combined_expr = f"({filter_expr}) AND ({expr})"
        elif filter_expr:
            combined_expr = filter_expr
        elif expr:
            combined_expr = expr
        else:
            combined_expr = ""
        # 8. 创建混合搜索请求
        hybrid_search_req = create_hybrid_search_requests(
            dense_vector=embed_vector['dense'][0],
            sparse_vector=embed_vector['sparse'][0],
            expr=combined_expr,
            expr_params=expr_params,
            limit=getattr(self.config, 'hybrid_search_limit', 10)
        )

        # 9. 执行混合搜索
        try:
            hybrid_search_res = execute_hybrid_search_query(
                milvus_client=milvus_client,
                collection_name=self.config.chunks_collection,
                search_requests=hybrid_search_req,
                output_fields=[
                    'chunk_id',
                    'content',
                    'book_name',
                    'title',
                    'content_type',
                    'author_name',
                    'source_file',
                    'category_tags'
                ],
            )
        except Exception as e:
            self.logger.error(f"混合搜索查询失败, 原因:{str(e)}")
            return {"embedding_chunks": []}

        if not hybrid_search_res or not hybrid_search_res[0]:
            self.logger.info("混合检索无结果")
            return {"embedding_chunks": []}

        state['embedding_chunks'] = hybrid_search_res[0]
        self.logger.info(f"混合检索完成，返回 {len(hybrid_search_res[0])} 条结果")

        return {"embedding_chunks": hybrid_search_res[0]}

    def _validate_state(self, state: QueryGraphState) -> Tuple[str, List[str]]:
        """校验state中的必要参数"""
        # 获取重写后的查询
        rewritten_query = state.get('rewritten_query', '')
        if not rewritten_query or not isinstance(rewritten_query, str):
            raise StateFieldError(
                node_name=self.name,
                field_name='rewritten_query',
                expected_type=str,
                message='rewritten_query不能为空'
            )

        # 获取确认的书名列表
        book_names = state.get('book_names', [])
        if not isinstance(book_names, list):
            book_names = []

        return rewritten_query, book_names

