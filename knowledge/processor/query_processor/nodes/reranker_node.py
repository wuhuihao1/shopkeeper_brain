from typing import Tuple, List, Dict, Any
import math
from FlagEmbedding import FlagReranker
from knowledge.processor.query_processor.base import BaseNode, T
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.utils.client.ai_clients import AIClients

class RerankerNode(BaseNode):
    name =  'reranker_node'
    @staticmethod
    def _sigmoid(score: float) -> float:
        """归一化函数,将 (-∞, +∞) 映射到 (0, 1)"""
        return 1.0 / (1.0 + math.exp(-score))

    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 获取用户问题
        user_query = state.get('rewritten_query') or state.get('original_query')
        # 获取两路检索结果(本地检索结果、远程检索结果)
        refine_docs: List[Dict[str, Any]] = self._collect_rerank_inputs(state)
        # 利用reanker进行精排
        refine_docs: List[Dict[str, Any]] = self._refine_rank(user_query, refine_docs)
        #通过断崖点切分文档
        reranked_docs = self._cliff_cutoff(refine_docs, self.config.rerank_min_top_k, self.config.rerank_max_top_k)
        # 保存在state中
        state['reranked_docs'] = reranked_docs
        return state

    def _cliff_cutoff(self, refine_docs: List[Dict[str, Any]], rerank_min_top_k: int, rerank_max_top_k: int) -> List[Dict[str, Any]]:
        """
        动态top_k: 归一化后只需一个断崖阈值(rerank_gap_threshold)
        从头开始寻找最大断崖点，再用 min_top_k 兜底
        """
        upper_bound = min(rerank_max_top_k, len(refine_docs))
        lower_bound = min(rerank_min_top_k, upper_bound)
        #标记切分点
        cut_off = upper_bound
        max_gap = 0
        # 从第0个开始,找最大断崖
        for i in range(0, upper_bound - 1):
            current_score = refine_docs[i]['score']
            next_score = refine_docs[i + 1]['score']
            # 判空
            if not current_score or not next_score:
                continue
            #计算分差
            gap = current_score - next_score

            if gap >= .15 and gap > max_gap:
                max_gap = gap
                cut_off = i + 1
        self.logger.info(f'位置{cut_off}发生断崖')
        # 如果断崖位置小于min_top_k依然保留Min个
        cut_off = max(cut_off, lower_bound)
        #使用切分点切分文档
        cutoff_docs = refine_docs[:cut_off]

        return cutoff_docs




    def _refine_rank(self, user_query: Any, refine_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        rerank模型进行打分,精排
        Args:
            user_query: 用户的查询
            refine_docs: 本地和远程融合后的检索结果
        Returns:
            Dict[str,Any]:{"score","","other":"..."}
        """
        if not refine_docs:
            return []
        try:
            rerank_client = AIClients.get_bge_m3_rerank_client()
        except ConnectionError as e:
            self.logger.error(f'连接到重排序模型有误, 原因{str(e)}')
            return [{ **doc, "score": None } for doc in refine_docs]
        #  构建问题->文档的pair对
        query_doc_pairs = [(user_query, doc['content']) for doc in refine_docs]
        # 使用重排序模型,对pair对进行打分
        rerank_scores = rerank_client.compute_score(sentence_pairs=query_doc_pairs)
        # 打分后得到一个浮点数组成的list
        # 组合score到原有的数据中
        doc_score = [{**doc, 'score': self._sigmoid(float(score)) }for doc, score in  list(zip(refine_docs, rerank_scores))]
        # 根据分数排序
        sorted_doc_score = sorted(doc_score, key=lambda x: x['score'], reverse=True)

        return sorted_doc_score

    def _collect_rerank_inputs(self, state: QueryGraphState)->List[Dict[str, Any]]:
        final_docs = []
        # rrf融合排序后得到的是chunks组成的列表
        rrf_chunks = state['rrf_chunks'] or []
        for chunk in rrf_chunks:
            #判空
            if not chunk or not isinstance(chunk, dict):
                continue
            content = chunk['content']
            if not content:
                continue
            # 获取title和id
            title = chunk['title']
            chunk_id = chunk['chunk_id']
            # 文档格式化
            formated_local_doc = self._format_doc(content=content, chunk_id=chunk_id, title=title, source='local')
            final_docs.append(formated_local_doc)
        # 获取远程检索结果
        web_search_docs = state['web_search_docs'] or []
        for doc in web_search_docs:
            #判空
            if not doc or not isinstance(doc, dict):
                continue
            snippet = doc['snippet']
            title = doc['title']
            url = doc['url']
            # 格式化
            formated_web_doc = self._format_doc(content=snippet, title=title,url=url, source='web')
            final_docs.append(formated_web_doc)
        self.logger.info(f'进入rerank搜索的数据总共{len(final_docs)}条')
        return final_docs


    def _format_doc(self, content: str, chunk_id: int= None, title: str = '',url: str ='', source: str = ''):
        """
        格式化本地以及远程检索到的文档
        Args:
            content: chunk包含内容
            chunk_id: chunk_id
            title: 标题
            url: web_search提供的url
            scorce: 数据源区分标识
        Returns:
        """
        return {
            'content': content,
            'chunk_id': chunk_id,
            'title': title,
            'url': url,
            'source': source,
        }

if __name__ == "__main__":
    print("=" * 60)
    print("开始测试: 重排序节点 (RerankNode)")
    print("=" * 60)

    mock_state = {
        "rewritten_query": "怎么测这块主板的短路问题？",
        "rrf_chunks": [
            {"chunk_id": "local_1", "title": "主板维修手册",
             "content": "主板短路通常表现为通电后风扇转一下就停，可以使用万用表的蜂鸣档测量。"},
            {"chunk_id": "local_2", "title": "闲聊",
             "content": "今天中午去吃猪脚饭吧，这块主板外观很漂亮。"},
        ],
        "web_search_docs": [
            {"url": "https://example.com/repair", "title": "短路查修指南",
             "snippet": "主板通电前先打各主供电电感的对地阻值，阻值偏低就是短路。"},
            {"url": "https://example.com/news", "title": "科技新闻",
             "snippet": "苹果发布新款手机，A系列芯片性能提升20%。"},
        ],
    }

    print("【输入状态】:")
    print(f"  查询: {mock_state['rewritten_query']}")
    print(f"  本地文档: {len(mock_state['rrf_chunks'])} 篇")
    print(f"  网络文档: {len(mock_state['web_search_docs'])} 篇")
    print("-" * 60)

    node = RerankerNode()
    result = node.process(mock_state)
