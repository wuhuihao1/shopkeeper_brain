from typing import Tuple, List, Dict, Any
from knowledge.processor.query_processor.base import BaseNode
from knowledge.processor.query_processor.state import QueryGraphState


class RrfMergeNode(BaseNode):
    """
    RRF 融合节点（听书平台版本）
    融合多路检索结果：
    1. embedding_chunks（向量检索）
    2. hyde_embedding_chunks（HyDE检索）
    3. web_search_docs（联网搜索）
    使用倒数排名融合算法
    """
    name = "rrf_merge_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 1. 获取多路检索结果
        embedding_chunks = state.get('embedding_chunks', []) or []
        hyde_embedding_chunks = state.get('hyde_embedding_chunks', []) or []
        web_search_docs = state.get('web_search_docs', []) or []

        # 2. 验证并提取实体
        valid_embedding = self._validate_search_result(embedding_chunks)
        valid_hyde = self._validate_search_result(hyde_embedding_chunks)
        valid_web = self._validate_web_search_result(web_search_docs)

        # 3. 配置各路检索结果的权重
        # 权重说明：向量检索和HyDE权重较高，联网搜索权重较低（作为补充）
        search_result_weight = {
            'embedding_search': (valid_embedding, getattr(self.config, 'rrf_weight_embedding', 1.0)),
            'hyde_search': (valid_hyde, getattr(self.config, 'rrf_weight_hyde', 1.0)),
            'web_search': (valid_web, getattr(self.config, 'rrf_weight_web', 0.5)),
        }

        # 4. 收集有效的搜索结果和权重
        rrf_inputs = []
        for name, (results, weight) in search_result_weight.items():
            if results:
                rrf_inputs.append((results, weight))
                self.logger.info(f"{name}: {len(results)} 条结果, 权重: {weight}")

        if not rrf_inputs:
            self.logger.warning("所有检索结果均为空")
            state['rrf_chunks'] = []
            return state

        # 5. RRF融合
        rrf_k = getattr(self.config, 'rrf_k', 60)
        rrf_max_results = getattr(self.config, 'rrf_max_results', 10)

        merged_results = self._merge_rrf_docs(rrf_inputs, rrf_k, rrf_max_results)

        # 6. 提取chunk数据
        merged_chunks = [chunk for chunk, _ in merged_results]

        state['rrf_chunks'] = merged_chunks
        self.logger.info(f"RRF融合完成，共 {len(merged_chunks)} 条结果")

        return state

    def _merge_rrf_docs(self, rrf_inputs: List[Tuple[List[Dict[str, Any]], float]],
                        rrf_k: int, rrf_max_results: int) -> List[Tuple[Dict[str, Any], float]]:
        """
        RRF计算多路检索返回的文档得分
        公式：weight(i) / (k + rank(i))
        Args:
            rrf_inputs: 多路检索的文档列表以及对应的权重
            rrf_k: 平滑参数
            rrf_max_results: 返回最大个数
        Returns:
            多路检索文档对象以及经过RRF计算之后的文档得分
        """
        chunk_score = {}
        chunk_data = {}

        for search_results, weight in rrf_inputs:
            for rank, entity in enumerate(search_results, 1):
                # 获取唯一标识（本地用chunk_id，联网用url）
                chunk_id = entity.get('chunk_id')
                url = entity.get('url')

                if chunk_id:
                    doc_id = f"local_{chunk_id}"
                elif url:
                    doc_id = f"web_{url}"
                else:
                    continue

                # RRF打分：weight / (k + rank)
                chunk_score[doc_id] = chunk_score.get(doc_id, 0.0) + weight / (rrf_k + rank)

                # 存储文档对象（只存第一次遇到的）
                if doc_id not in chunk_data:
                    chunk_data[doc_id] = entity

        # 排序
        sorted_results = sorted(
            [(chunk_data[doc_id], score) for doc_id, score in chunk_score.items()],
            key=lambda x: x[1],
            reverse=True
        )

        return sorted_results[:rrf_max_results] if rrf_max_results else sorted_results

    def _validate_search_result(self, search_result: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        验证本地检索结果，提取entity
        Args:
            search_result: Milvus返回的搜索结果
        Returns:
            提取后的文档列表
        """
        if not search_result:
            return []

        validated = []
        for res in search_result:
            if not res or not isinstance(res, dict):
                continue

            # 提取entity（Milvus返回格式）
            entity = res.get('entity')
            if entity and isinstance(entity, dict):
                validated.append(entity)
            else:
                # 如果已经是entity格式，直接使用
                validated.append(res)

        return validated

    def _validate_web_search_result(self, web_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        验证联网搜索结果，统一格式
        Args:
            web_results: 联网搜索返回的结果
        Returns:
            统一格式后的文档列表
        """
        if not web_results:
            return []

        validated = []
        for res in web_results:
            if not res or not isinstance(res, dict):
                continue

            # 确保有必要的字段
            if res.get('snippet') or res.get('title'):
                # 添加 content 字段（与本地检索对齐）
                if 'content' not in res:
                    res['content'] = res.get('snippet', '')
                validated.append(res)

        return validated


if __name__ == '__main__':
    # 测试代码（config 由 BaseNode 自动提供）
    node = RrfMergeNode()

    # 模拟测试数据
    mock_state = {
        "embedding_chunks": [
            {"entity": {"chunk_id": 1, "content": "向量检索结果1", "book_name": "活着", "score": 0.85}},
            {"entity": {"chunk_id": 2, "content": "向量检索结果2", "book_name": "活着", "score": 0.75}},
        ],
        "hyde_embedding_chunks": [
            {"entity": {"chunk_id": 2, "content": "HyDE结果1", "book_name": "活着", "score": 0.80}},
            {"entity": {"chunk_id": 3, "content": "HyDE结果2", "book_name": "活着", "score": 0.70}},
        ],
        "web_search_docs": [
            {"url": "https://example.com/1", "title": "网页1", "snippet": "联网搜索结果1", "content": "联网搜索结果1"},
            {"url": "https://example.com/2", "title": "网页2", "snippet": "联网搜索结果2", "content": "联网搜索结果2"},
        ]
    }

    result = node.process(mock_state)

    chunks = result.get('rrf_chunks', [])
    print(f"\n{'=' * 60}")
    print(f"RRF融合结果: {len(chunks)} 条")
    print(f"{'=' * 60}")

    for i, chunk in enumerate(chunks, 1):
        print(f"\n[{i}] {chunk}")