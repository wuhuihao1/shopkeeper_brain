from typing import List, Dict, Any, Tuple
import math
from FlagEmbedding import FlagReranker
from knowledge.processor.query_processor.base import BaseNode
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.utils.client.ai_clients import AIClients


class RerankerNode(BaseNode):
    """
    重排序节点（听书平台版本）
    1. 收集本地检索和联网检索结果
    2. 使用 BGE-M3 Reranker 模型进行精排
    3. 动态断崖切分，选择最佳文档
    """
    name = 'reranker_node'

    @staticmethod
    def _sigmoid(score: float) -> float:
        """归一化函数，将 (-∞, +∞) 映射到 (0, 1)"""
        return 1.0 / (1.0 + math.exp(-score))

    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 获取用户问题
        user_query = state.get('rewritten_query') or state.get('original_query')

        # 收集需要重排序的文档
        refine_docs = self._collect_rerank_inputs(state)

        if not refine_docs:
            self.logger.warning("没有需要重排序的文档")
            state['reranked_docs'] = []
            return state

        # 使用reranker进行精排
        refine_docs = self._refine_rank(user_query, refine_docs)

        # 通过断崖点切分文档
        rerank_min_top_k = getattr(self.config, 'rerank_min_top_k', 3)
        rerank_max_top_k = getattr(self.config, 'rerank_max_top_k', 10)
        reranked_docs = self._cliff_cutoff(refine_docs, rerank_min_top_k, rerank_max_top_k)

        # 保存到state中
        state['reranked_docs'] = reranked_docs
        self.logger.info(f"重排序完成，最终保留 {len(reranked_docs)} 条文档")

        return state

    def _cliff_cutoff(self, refine_docs: List[Dict[str, Any]],
                      rerank_min_top_k: int,
                      rerank_max_top_k: int) -> List[Dict[str, Any]]:
        """
        动态top_k: 寻找最大断崖点进行切分
        Args:
            refine_docs: 重排序后的文档列表（已按分数降序）
            rerank_min_top_k: 最小保留数量
            rerank_max_top_k: 最大保留数量
        Returns:
            切分后的文档列表
        """
        if not refine_docs:
            return []

        upper_bound = min(rerank_max_top_k, len(refine_docs))
        lower_bound = min(rerank_min_top_k, upper_bound)

        cut_off = upper_bound
        max_gap = 0
        gap_threshold = getattr(self.config, 'rerank_gap_threshold', 0.15)

        # 寻找最大断崖点
        for i in range(0, upper_bound - 1):
            current_score = refine_docs[i].get('score')
            next_score = refine_docs[i + 1].get('score')

            if current_score is None or next_score is None:
                continue

            gap = current_score - next_score

            if gap >= gap_threshold and gap > max_gap:
                max_gap = gap
                cut_off = i + 1
                self.logger.info(f"发现断崖: 位置 {cut_off}, 分差 {gap:.4f}")

        # 确保不少于最小保留数量
        cut_off = max(cut_off, lower_bound)

        return refine_docs[:cut_off]

    def _refine_rank(self, user_query: str, refine_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        rerank模型进行打分精排
        Args:
            user_query: 用户查询
            refine_docs: 待重排序的文档列表
        Returns:
            重排序后的文档列表（已按分数降序）
        """
        if not refine_docs:
            return []

        # 获取reranker客户端
        try:
            rerank_client = AIClients.get_bge_m3_rerank_client()
        except ConnectionError as e:
            self.logger.error(f'连接到重排序模型失败, 原因{str(e)}')
            # 降级：返回原顺序，score设为None
            return [{**doc, "score": None} for doc in refine_docs]

        # 构建问题-文档对
        query_doc_pairs = [(user_query, doc['content']) for doc in refine_docs]

        # 使用重排序模型打分
        try:
            rerank_scores = rerank_client.compute_score(sentence_pairs=query_doc_pairs)
        except Exception as e:
            self.logger.error(f'重排序打分失败, 原因{str(e)}')
            return [{**doc, "score": None} for doc in refine_docs]

        # 组合分数并归一化
        doc_score = []
        for doc, score in zip(refine_docs, rerank_scores):
            normalized_score = self._sigmoid(float(score))
            doc_score.append({**doc, 'score': normalized_score})

        # 按分数降序排序
        sorted_doc_score = sorted(doc_score, key=lambda x: x['score'], reverse=True)

        return sorted_doc_score

    def _collect_rerank_inputs(self, state: QueryGraphState) -> List[Dict[str, Any]]:
        """
        收集需要重排序的文档
        包括：
        1. RRF融合后的本地chunks
        2. 联网搜索结果
        """
        final_docs = []

        # 1. 收集RRF融合后的本地chunks
        rrf_chunks = state.get('rrf_chunks', []) or []
        for chunk in rrf_chunks:
            if not chunk or not isinstance(chunk, dict):
                continue

            content = chunk.get('content', '')
            if not content:
                continue

            # 格式化本地文档
            formatted_doc = self._format_local_doc(chunk)
            final_docs.append(formatted_doc)

        # 2. 收集联网搜索结果
        web_search_docs = state.get('web_search_docs', []) or []
        for doc in web_search_docs:
            if not doc or not isinstance(doc, dict):
                continue

            content = doc.get('content', '') or doc.get('snippet', '')
            if not content:
                continue

            # 格式化联网文档
            formatted_doc = self._format_web_doc(doc)
            final_docs.append(formatted_doc)

        self.logger.info(f'进入重排序的文档总数: {len(final_docs)} 条')
        return final_docs

    def _format_local_doc(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """
        格式化本地检索到的文档
        Args:
            chunk: 本地chunk数据
        Returns:
            格式化后的文档
        """
        return {
            'content': chunk.get('content', ''),
            'chunk_id': chunk.get('chunk_id'),
            'title': chunk.get('title', ''),
            'book_name': chunk.get('book_name', ''),
            'author_name': chunk.get('author_name', ''),
            'content_type': chunk.get('content_type', ''),
            'source': 'local',
            'source_file': chunk.get('source_file', ''),
            'category_tags': chunk.get('category_tags', '[]'),
        }

    def _format_web_doc(self, doc: Dict[str, Any]) -> Dict[str, Any]:
        """
        格式化联网搜索到的文档
        Args:
            doc: 联网搜索文档数据
        Returns:
            格式化后的文档
        """
        return {
            'content': doc.get('content', '') or doc.get('snippet', ''),
            'title': doc.get('title', ''),
            'url': doc.get('url', ''),
            'source': 'web',
            'content_type': 'web_search',
            'chunk_id': None,
            'book_name': '',
            'author_name': '',
            'source_file': '',
            'category_tags': '[]',
        }


if __name__ == '__main__':
    # 测试代码（config 由 BaseNode 自动提供）
    node = RerankerNode()

    # 模拟测试数据
    mock_state = {
        "rewritten_query": "这本书的主人公经历了哪些苦难？",
        "rrf_chunks": [
            {
                "chunk_id": 1,
                "content": "福贵经历了儿子有庆献血致死、女儿凤霞难产而死、妻子家珍病逝、女婿二喜被砸死、孙子苦根吃豆子噎死等一系列悲剧。",
                "title": "听书笔记",
                "book_name": "活着",
                "author_name": "余华",
                "content_type": "听书笔记",
                "source_file": "活着_听书笔记.md"
            },
            {
                "chunk_id": 2,
                "content": "这是一本关于普通人在苦难中生存的小说，语言朴素但情感深刻。",
                "title": "书籍简介",
                "book_name": "活着",
                "author_name": "余华",
                "content_type": "书籍简介",
                "source_file": "活着_简介.md"
            }
        ],
        "web_search_docs": [
            {
                "url": "https://example.com/review",
                "title": "《活着》书评",
                "snippet": "福贵的一生充满了失去，但他依然坚强地活着。这部作品让人思考生命的意义。",
                "content": "福贵的一生充满了失去，但他依然坚强地活着。这部作品让人思考生命的意义。"
            }
        ]
    }

    result = node.process(mock_state)

    docs = result.get('reranked_docs', [])
    print(f"\n{'='*60}")
    print(f"重排序结果: {len(docs)} 条")
    print(f"{'='*60}")

    for i, doc in enumerate(docs, 1):
        print(f"\n[{i}] 得分: {doc.get('score'):.4f}" if doc.get('score') else f"\n[{i}] 得分: None")
        print(f"    来源: {doc.get('source')}")
        print(f"    书名: {doc.get('book_name')}")
        print(f"    类型: {doc.get('content_type')}")
        print(f"    内容: {doc.get('content', '')[:100]}...")