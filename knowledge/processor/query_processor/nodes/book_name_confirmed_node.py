import logging
import re
import json
from json import JSONDecodeError
from typing import Dict, Tuple, List, Any

from langchain_core.messages import SystemMessage, HumanMessage

from knowledge.processor.query_processor.base import BaseNode
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.processor.query_processor.exceptions import StateFieldError
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.prompts.query_prompt import BOOK_NAME_USER_EXTRACT_TEMPLATE
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
from knowledge.utils.milvus_util import create_hybrid_search_requests, execute_hybrid_search_query
from knowledge.processor.query_processor.base import get_config
from knowledge.utils.mongo_history_util import get_recent_messages


class _BookNameExtractor:
    """书名提取器：从用户查询中提取书名并重写查询"""

    def extract_book_name(self, origin_query: str, history_context: str) -> Dict[str, Any]:
        """
        通过LLM提取书名并重写查询
        Args:
            origin_query: 用户的原始查询
            history_context: 历史上下文
        Returns:
            包含 book_names 和 rewritten_query 的字典
        """
        # 默认结果
        llm_result = {"book_names": [], "rewritten_query": origin_query}

        # 获取LLM客户端
        try:
            llm_client = AIClients.get_llm_client(response_format=True)
        except ConnectionError as e:
            logging.error(f'LLM连接失败, 原因{str(e)}')
            return llm_result

        # 构建提示词
        book_name_system_prompt = '您是一位书籍识别专家，请从用户的问题以及历史对话中提取相关的书名并改写原始查询内容'
        book_name_user_prompt = BOOK_NAME_USER_EXTRACT_TEMPLATE.format(
            history_text=history_context,
            query=origin_query,
        )

        # 调用LLM
        try:
            llm_res = llm_client.invoke([
                SystemMessage(content=book_name_system_prompt),
                HumanMessage(content=book_name_user_prompt),
            ])
        except ConnectionError as e:
            logging.error(f'LLM调用失败,原因{str(e)}')
            return llm_result

        llm_content = llm_res.content
        if not llm_content:
            return llm_result

        # 解析结果
        parsed_result = self._clean_and_parse(llm_content)
        llm_result['book_names'] = parsed_result.get('book_names', [])
        llm_result['rewritten_query'] = parsed_result.get('rewritten_query', origin_query)

        logging.info(f"LLM提取结果 - 原查询: {origin_query}, 提取书名: {llm_result['book_names']}, 重写: {llm_result['rewritten_query']}")
        return llm_result

    def _clean_and_parse(self, llm_response_content: str) -> Dict[str, Any]:
        """
        清洗解析LLM的输出结果
        Args:
            llm_response_content: LLM输出内容
        Returns:
            解析后的字典
        """
        try:
            # 去除JSON代码块围栏
            cleaned = re.sub(r"^```(?:json)?\s*", "", llm_response_content.strip())
            content = re.sub(r"\s*```$", "", cleaned)

            # 反序列化
            llm_content_obj = json.loads(content)

            # 获取 book_names
            raw_book_names = llm_content_obj.get('book_names', [])
            if not isinstance(raw_book_names, list):
                book_names = []
            else:
                book_names = [name.strip() for name in raw_book_names if isinstance(name, str) and name.strip()]

            # 获取 rewritten_query
            raw_rewritten_query = llm_content_obj.get('rewritten_query', '')
            if not isinstance(raw_rewritten_query, str):
                rewritten_query = ""
            else:
                rewritten_query = raw_rewritten_query.strip()

            return {
                "book_names": book_names,
                "rewritten_query": rewritten_query
            }
        except JSONDecodeError as e:
            logging.error(f'JSON反序列化失败, 原因{str(e)}')
            raise JSONDecodeError(msg=e.msg, doc=e.doc, pos=e.pos)


class _BookNameAligner:
    """书名对齐器：将提取的书名与Milvus中的书名进行匹配"""

    def __init__(self):
        self._config = get_config()

    def search_and_align(self, book_names: List[str]) -> Tuple[List[str], List[str]]:
        """
        检索向量数据库并与库中的书名对齐
        Args:
            book_names: LLM提取到的书名列表
        Returns:
            (confirmed, options) - 确认的书名列表和候选书名列表
        """
        # 1. 混合检索向量数据库
        search_result = self._search_vector(book_names)
        if not search_result:
            return [], []

        # 2. 根据检索结果做对齐
        confirmed, options = self._align(search_result)

        # 3. 分数差异化过滤
        if len(confirmed) > 1:
            confirmed = self._book_name_score_filter(confirmed, search_result)

        logging.info(f"书名对齐结果 - 提取: {book_names}, 确认: {confirmed}, 候选: {options}")
        return confirmed, options

    def _book_name_score_filter(self, confirmed: List[str], search_result: List[Dict[str, Any]]) -> List[str]:
        """
        根据得分过滤确认的书名
        如果最高分与其他分数差距过大，只保留最高分
        """
        # 构建书名 -> 最大得分的映射
        book_name_score = {}
        for search_item in search_result:
            search_matches = search_item.get('matches', [])
            for item in search_matches:
                item_score = item.get('score', 0)
                item_name = item.get('book_name', '')
                if item_name in confirmed:
                    book_name_score[item_name] = max(book_name_score.get(item_name, 0), item_score)

        if not book_name_score:
            return confirmed

        # 计算最高分
        max_score = max(book_name_score.values())
        score_gap = getattr(self._config, 'book_name_score_gap', 0.15)

        # 过滤掉与最高分差距过大的书名
        return [name for name, score in book_name_score.items() if max_score - score <= score_gap]

    def _search_vector(self, book_names: List[str]) -> List[Dict[str, Any]]:
        """
        对提取到的所有书名进行向量化，然后去Milvus中检索
        """
        final_search_result = []

        # 获取Milvus客户端
        try:
            milvus_client = StorageClients.get_milvus_client()
        except ConnectionError as e:
            logging.error(f'连接Milvus失败, 原因{str(e)}')
            return final_search_result

        # 获取BGE模型
        try:
            bge_model = AIClients.get_bge_m3_client()
        except ConnectionError as e:
            logging.error(f'BGE模型获取失败, 原因{str(e)}')
            return final_search_result

        # 书名列表向量化
        try:
            hybrid_vector_result = generate_bge_m3_hybrid_vectors(model=bge_model, embedding_documents=book_names)
        except Exception as e:
            logging.error(f'书名列表{book_names}生成混合向量失败 原因:{str(e)}')
            return final_search_result

        # 获取集合名
        collection_name = getattr(self._config, 'book_name_collection', 'book_name_collection')

        # 混合向量检索
        for index, book_name in enumerate(book_names):
            # 构建混合检索请求
            hybrid_requests = create_hybrid_search_requests(
                dense_vector=hybrid_vector_result['dense'][index],
                sparse_vector=hybrid_vector_result['sparse'][index]
            )
            # 执行混合检索
            hybrid_search_result = execute_hybrid_search_query(
                milvus_client=milvus_client,
                collection_name=collection_name,
                search_requests=hybrid_requests,
                output_fields=['book_name']
            )
            # 解析检索结果
            matches = [{
                'score': search_item['distance'],
                'book_name': search_item['entity']['book_name']
            } for search_item in (hybrid_search_result[0] if hybrid_search_result else [])]

            final_search_result.append({
                "extracted_name": book_name,
                "matches": matches
            })

        return final_search_result

    def _align(self, search_result: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
        """
        对齐结果
        规则：
        1. 得分 > high_confidence -> confirmed
        2. mid_confidence < 得分 <= high_confidence -> options
        3. 得分 <= mid_confidence -> 丢弃
        """
        confirmed = []
        options = []

        high_threshold = getattr(self._config, 'book_name_high_confidence', 0.7)
        mid_threshold = getattr(self._config, 'book_name_mid_confidence', 0.45)
        score_gap = getattr(self._config, 'book_name_score_gap', 0.15)
        max_options = getattr(self._config, 'book_name_max_options', 3)

        def add_confirmed(item_name: str):
            if item_name not in confirmed:
                confirmed.append(item_name)

        for search_item in search_result:
            extracted_name = search_item['extracted_name']
            matches = search_item['matches']
            sorted_matches = sorted(matches, key=lambda x: x['score'], reverse=True)

            # 高置信度匹配
            high_score_items = [item for item in sorted_matches if item['score'] > high_threshold]

            if high_score_items:
                # 检查是否有与提取名称完全匹配的
                exact_match = next(
                    (item for item in high_score_items if item['book_name'] == extracted_name),
                    None
                )
                if exact_match:
                    add_confirmed(exact_match['book_name'])
                elif len(high_score_items) == 1:
                    add_confirmed(high_score_items[0]['book_name'])
                else:
                    # 多个高分结果
                    top_score = high_score_items[0]['score']
                    if top_score - high_score_items[1]['score'] >= score_gap:
                        add_confirmed(high_score_items[0]['book_name'])
                    else:
                        # 差距不大，放入options
                        for item in high_score_items[:max_options]:
                            if item['book_name'] not in confirmed:
                                options.append(item)
            else:
                # 中置信度匹配
                mid_score_items = [item for item in sorted_matches if item['score'] > mid_threshold]
                if mid_score_items:
                    added_count = 0
                    for item in mid_score_items:
                        if added_count >= max_options:
                            break
                        if item['book_name'] not in confirmed:
                            options.append(item)
                            added_count += 1

        # 去重并限制options数量
        unique_options = []
        seen = set()
        for opt in options:
            if opt['book_name'] not in seen and opt['book_name'] not in confirmed:
                seen.add(opt['book_name'])
                unique_options.append(opt)

        sorted_options = sorted(unique_options, key=lambda x: x['score'], reverse=True)
        options_names = [item['book_name'] for item in sorted_options[:max_options]]

        return confirmed, options_names


class BookNameConfirmedNode(BaseNode):
    """
    书名确认节点
    1. 利用LLM从用户查询中提取书名
    2. 向量化后去Milvus中检索匹配
    3. 根据匹配分数得到确认的书名和候选书名
    4. 决策：有确认则继续检索，有候选则反问用户，无结果则提示
    """
    name = "book_name_confirmed_node"

    def __init__(self):
        super().__init__()
        self._extractor = _BookNameExtractor()
        self._aligner = _BookNameAligner()

    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 1. 获取用户原始问题
        origin_query = state.get('original_query', '')
        if not origin_query:
            raise StateFieldError(
                node_name=self.name,
                field_name='original_query',
                expected_type=str,
                message='用户问题不能为空'
            )

        # 2. 获取历史对话
        session_id = state.get('session_id', '')
        history_context = get_recent_messages(session_id=session_id, limit=10)
        formatted_history = self._format_history_context(history_context)

        # 3. 利用LLM提取书名和重写查询
        llm_result = self._extractor.extract_book_name(origin_query, formatted_history)
        book_names = llm_result.get('book_names', [])
        rewritten_query = llm_result.get('rewritten_query', origin_query)

        # 4. 根据书名列表做对齐确认
        if book_names:
            confirmed, options = self._aligner.search_and_align(book_names)
        else:
            confirmed, options = [], []

        # 5. 决策并更新state
        self._decide(confirmed, options, state, rewritten_query)
        # 6. 保存历史
        state['history'] = history_context

        return state

    def _format_history_context(self, history_context: list) -> str:
        """格式化历史对话上下文"""
        if not history_context:
            return "无历史对话"

        formatted = []
        for msg in history_context:
            role = msg.get('role', '')
            text = msg.get('text', '')
            if role and text:
                formatted.append(f'角色:{role}, 内容:{text}')
        return " ".join(formatted)

    def _decide(self, confirmed: List[str], options: List[str], state: QueryGraphState, rewritten_query: str):
        """
        根据confirmed和options决策
        - confirmed有值：继续检索
        - options有值：反问用户确认
        - 都为空：提示无法识别
        """
        if confirmed:
            state['book_names'] = confirmed
            state['rewritten_query'] = rewritten_query
            self.logger.info(f"书名确认成功: {confirmed}, 重写查询: {rewritten_query}")
        elif options:
            state['answer'] = f"我不确定您指的是哪本书。您是在询问以下书籍吗：{'、'.join(options)}？"
            self.logger.info(f"书名候选: {options}, 等待用户确认")
        else:
            state['answer'] = '抱歉，我无法识别您询问的具体书籍名称，请提供更准确的书籍名称。'
            self.logger.warning(f"无法识别书名, 原始查询: {state.get('original_query', '')}")



