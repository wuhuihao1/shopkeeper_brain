from typing import Tuple, List, Dict, Any, Union
from knowledge.processor.query_processor.base import BaseNode, T
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.processor.query_processor.exceptions import StateFieldError
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
from knowledge.utils.milvus_util import create_hybrid_search_requests, execute_hybrid_search_query, _item_names_filter

class HybridVectorSearch(BaseNode):
    name = 'hybrid_vector_search_node'

    def process(self, state: QueryGraphState) -> Union[QueryGraphState,Dict[str,Any]]:
        rewritten_query, item_names = self._validate_state(state)

        # 2. 获取嵌入模型客户端
        try:
            bge_client = AIClients.get_bge_m3_client()
        except ConnectionError as e:
            self.logger.error(f'获取嵌入模型客户端失败, 原因{str(e)}')
            return {"embedding_chunks": []}
        # 3. 获取Milvus客户端
        try:
            milvus_client = StorageClients.get_milvus_client()
        except ConnectionError as e:
            self.logger.error(f'获取milvus客户端失败, 原因{str(e)}')
            return {"embedding_chunks": []}

        # 4. 对rewritten_query进行嵌入、检索
        try:
            embed_vector = generate_bge_m3_hybrid_vectors(model=bge_client, embedding_documents=[rewritten_query])
        except Exception as e:
            self.logger.error(f'用户问题{rewritten_query}嵌入向量获取失败, 原因{str(e)}')
            return {"embedding_chunks": []}
        # 创建expr和expr params进行过滤
        expr, expr_params = _item_names_filter(item_names)
        # 创建混合搜索请求向量场
        hybrid_search_req = create_hybrid_search_requests(
            dense_vector=embed_vector['dense'][0],
            sparse_vector=embed_vector['sparse'][0],
            expr=expr,
            expr_params=expr_params,
            limit=5
        )
        # 执行混合搜索请求
        hybrid_search_res = execute_hybrid_search_query(
            milvus_client=milvus_client,
            collection_name=self.config.chunks_collection,
            search_requests=hybrid_search_req,
            output_fields=['chunk_id', 'content', 'item_name', 'title']
        )

        if not hybrid_search_res or not hybrid_search_res[0]:
            return {"embedding_chunks": []}
        state['embedding_chunks'] = hybrid_search_res[0]
        return {"embedding_chunks": hybrid_search_res[0]}

    def _validate_state(self, state: QueryGraphState) -> Tuple[str, List[str]]:
        # 1. 用户的问题（LLM重写后的）
        rewritten_query = state['rewritten_query']
        # 2. 获取商品名列表
        item_names = state['item_names']
        # 3. 校验非空及类型
        if not rewritten_query or not isinstance(rewritten_query, str):
            raise StateFieldError(node_name=self.name, field_name='rewritten_query', expected_type=str)
        if not item_names or not isinstance(item_names, list):
            raise StateFieldError(node_name=self.name, field_name='item_names', expected_type=list)

        return rewritten_query, item_names

if __name__ == '__main__':
    import json

    state = {
        "rewritten_query": "万用表如何测量电阻",
        "item_names": ["RS-12 数字万用表"]  # 没用的
    }

    vector_search = HybridVectorSearch()
    result = vector_search.process(state)

    for r in result.get('embedding_chunks'):
        print(json.dumps(r, ensure_ascii=False, indent=2))