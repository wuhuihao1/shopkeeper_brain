from typing import Tuple, List, Dict, Any, Optional, Union

from langchain_core.messages import SystemMessage, HumanMessage

from knowledge.processor.query_processor.base import BaseNode
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.processor.query_processor.exceptions import StateFieldError
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.prompts.query_prompt import HYDE_SYSTEM_PROMPT_BOOK, HYDE_USER_PROMPT_TEMPLATE_BOOK
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
from knowledge.utils.milvus_util import create_hybrid_search_requests, execute_hybrid_search_query, book_names_filter


class HyDeVectorSearchNode(BaseNode):
    """
    HyDE 向量检索节点（听书平台版本）
    1. 利用LLM生成假设性文档（模拟书籍介绍/听书笔记风格）
    2. 将假设性文档向量化
    3. 在Milvus中进行混合检索
    """
    name = "hyde_vector_search_node"

    def process(self, state: QueryGraphState) -> Union[QueryGraphState, Dict[str, Any]]:
        # 1. 参数校验
        rewritten_query, book_names = self._validate_state(state)

        # 2. 获取过滤表达式（从 MetadataFilterNode 构建）
        filter_expr = state.get('filter_expr', '')

        # 3. 利用LLM生成假设性文档
        hy_document = self._generate_hy_document(rewritten_query, book_names)

        if not hy_document:
            self.logger.warning("生成假设性文档失败，返回空结果")
            return {"hyde_embedding_chunks": []}

        # 4. 获取嵌入模型客户端
        try:
            bge_client = AIClients.get_bge_m3_client()
        except ConnectionError as e:
            self.logger.error(f'获取嵌入模型客户端失败, 原因{str(e)}')
            return {"hyde_embedding_chunks": []}

        # 5. 获取Milvus客户端
        try:
            milvus_client = StorageClients.get_milvus_client()
        except ConnectionError as e:
            self.logger.error(f'获取milvus客户端失败, 原因{str(e)}')
            return {"hyde_embedding_chunks": []}

        # 6. 为假设性文档生成向量
        try:
            hy_vector = generate_bge_m3_hybrid_vectors(
                model=bge_client,
                embedding_documents=[hy_document]
            )
        except Exception as e:
            self.logger.error(f'假设性文档向量化失败, 原因{str(e)}')
            return {"hyde_embedding_chunks": []}

        # 7. 构建过滤表达式
        expr, expr_params = book_names_filter(book_names)

        # 如果已有过滤表达式，合并
        if filter_expr and expr:
            combined_expr = f"({filter_expr}) AND ({expr})"
        elif filter_expr:
            combined_expr = filter_expr
        elif expr:
            combined_expr = expr
        else:
            combined_expr = ""

        # 8. 构建混合检索请求
        hybrid_search_request = create_hybrid_search_requests(
            dense_vector=hy_vector['dense'][0],
            sparse_vector=hy_vector['sparse'][0],
            expr=combined_expr,
            expr_params=expr_params,
            limit=getattr(self.config, 'hyde_search_limit', 10)
        )
        # 9. 执行检索
        try:
            hybrid_search_res = execute_hybrid_search_query(
                milvus_client=milvus_client,
                collection_name=self.config.chunks_collection,
                search_requests=hybrid_search_request,
                output_fields=[
                    'chunk_id',
                    'book_name',
                    'content',
                    'title',
                    'content_type',
                    'author_name',
                    'source_file',
                    'category_tags'
                ],
            )
        except Exception as e:
            self.logger.error(f"HyDE混合搜索查询失败, 原因:{str(e)}")
            return {"hyde_embedding_chunks": []}

        if not hybrid_search_res or not hybrid_search_res[0]:
            self.logger.info("HyDE检索无结果")
            return {"hyde_embedding_chunks": []}
        state['hyde_embedding_chunks'] = hybrid_search_res[0]
        self.logger.info(f"HyDE检索完成，返回 {len(hybrid_search_res[0])} 条结果")

        return {"hyde_embedding_chunks": hybrid_search_res[0]}

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
        if not book_names or not isinstance(book_names, list):
            book_names = []

        return rewritten_query, book_names

    def _generate_hy_document(self, rewritten_query: str, book_names: List[str]) -> Optional[str]:
        """
        利用LLM生成假设性文档
        模拟书籍介绍、听书笔记、内容摘要的风格
        Args:
            rewritten_query: 重写后的查询
            book_names: 确认的书名列表
        Returns:
            生成的假设性文档
        """
        # 获取LLM客户端
        try:
            llm_client = AIClients.get_llm_client(response_format=False)
        except ConnectionError as e:
            self.logger.error(f'LLM获取失败, 原因:{str(e)}')
            return None

        # 构建提示词（从外部导入）
        book_names_str = '、'.join(book_names) if book_names else '相关书籍'
        system_prompt = HYDE_SYSTEM_PROMPT_BOOK.format(book_names=book_names_str)
        user_prompt = HYDE_USER_PROMPT_TEMPLATE_BOOK.format(
            book_names=book_names_str,
            rewritten_query=rewritten_query
        )

        # 调用LLM
        try:
            llm_result = llm_client.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ])
        except Exception as e:
            self.logger.error(f'LLM生成假设性文档失败, 原因:{str(e)}')
            return None

        # 判断是否有内容
        content = llm_result.content.strip() if llm_result.content else None
        if content:
            self.logger.info(f"HyDE假设性文档生成成功，长度: {len(content)} 字符")

        return content


if __name__ == "__main__":
    # 测试代码



    mock_state = {
        "rewritten_query": "这本书的主人公经历了哪些苦难？",
        "book_names": ["活着"],
        "filter_expr": '(book_name == "活着") AND (content_type in ["书籍简介", "听书笔记"])'
    }

    node = HyDeVectorSearchNode()

    result = node.process(mock_state)

    chunks = result.get("hyde_embedding_chunks", [])
    print(f"\n{'=' * 60}")
    print(f"HyDE 检索结果: {len(chunks)} 条")
    print(f"{'=' * 60}")

    for i, chunk in enumerate(chunks, 1):
        entity = chunk.get("entity", {})
        print(f"\n[{i}] 得分: {chunk.get('distance', 'N/A'):.4f}")
        print(f"    书名: {entity.get('book_name')}")
        print(f"    类型: {entity.get('content_type')}")
        print(f"    内容: {entity.get('content', '')[:100]}...")