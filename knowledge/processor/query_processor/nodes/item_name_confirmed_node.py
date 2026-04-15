import logging, re, json
from json import JSONDecodeError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
from typing import Dict, Tuple, List, Any
from langchain_core.messages import SystemMessage, HumanMessage
from knowledge.processor.query_processor.base import BaseNode
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.prompts.query_prompt import ITEM_NAME_USER_EXTRACT_TEMPLATE
from knowledge.utils.embedding_util import generate_bge_m3_hybrid_vectors
from knowledge.utils.milvus_util import create_hybrid_search_requests, execute_hybrid_search_query
from knowledge.processor.query_processor.base import get_config
from knowledge.utils.mongo_history_util import get_recent_messages

class _ItemNameExtractor:
    def extract_item_name(self, origin_query: str, history_context: str) -> Dict[str, Any]:
        """
        通过LLM大模型提取商品名并重写查询
        Args:
            origin_query: 用户的原始查询
            history_context: 历史上下文
        Returns:
        """
        # 1. 定义LLM输出默认结果
        llm_result = { "item_name": [], "rewritten_query": origin_query }

        # 2. 获取LLM客户端
        try:
            llm_client = AIClients.get_llm_client(response_format=True)
        except ConnectionError as e:
            logger.error(f'LLM连接失败, 原因{str(e)}')
            return llm_result
        # 3. 获取商品名提取的提示词
        # 3.1 系统提示词
        item_name_system_prompt = '您是一位商品提示词专家,请从用户的问题以及历史对话中提取相关的商品名并改写原始查询内容'
        # 3.2 用户提示词
        item_name_user_prompt = ITEM_NAME_USER_EXTRACT_TEMPLATE.format(
            history_text=history_context,
            query=origin_query,
        )
        # 4. 调用LLM
        try:
            llm_res  = llm_client.invoke(
                [
                    SystemMessage(content=item_name_system_prompt),
                    HumanMessage(content=item_name_user_prompt),
                ]
            )
        except ConnectionError as e:
            logger.error(f'LLM调用失败,原因{str(e)}')
            return llm_result
        llm_content = llm_res.content

        if not llm_content:
            return llm_result

        # 5. 清洗数据
        parsed_result: Dict[str, Any] = self._clean_and_parse(llm_content)

        llm_result['item_name'] = parsed_result.get('item_name', '')
        llm_result['rewritten_query'] = parsed_result.get('rewritten_query', origin_query)
        return llm_result


    def _clean_and_parse(self, llm_response_content: str) -> Dict[str, Any]:
        """
        清洗解析llm的输出结果
        Args:
            llm_content: llm输出
        """
        try:
            # 去除Json代码块围栏
            cleaned = re.sub(r"^```(?:json)?\s*", "", llm_response_content.strip())
            content = re.sub(r"\s*```$", "", cleaned)

            # 反序列化
            # llm_content_obj格式:
            """
                {
                    "item_names": ["商品A", "商品B"],
                    "rewritten_query": "改写后的问题"
                }
                """
            llm_content_obj: List[Dict[str, Any]] = json.loads(content)

            raw_item_names = llm_content_obj.get('item_names', '')

            # 判断类型
            if not isinstance(raw_item_names, list):
                item_names = []
            else:
                item_names =[item_name.strip() for item_name in raw_item_names if isinstance(item_name, str) and item_name.strip()]

            raw_rewritten_query = llm_content_obj.get('rewritten_query', '')
            if not isinstance(raw_rewritten_query, str):
                rewritten_query = ""
            else:
                rewritten_query = raw_rewritten_query.strip()

            return {
                "item_name": item_names,
                "rewritten_query": rewritten_query
            }
        except JSONDecodeError as e:
            logger.error(f'JSON反序列化失败, 原因{str(e)}')
            raise JSONDecodeError(
                msg=e.msg,
                doc=e.doc,
                pos=e.pos,
            )

class _ItemNameAligner:
    def __init__(self):
        self._config = get_config()

    def search_and_align(self, item_names: List[str]) -> Tuple[List[str], List[str]]:
        """
        检索向量数据库并且和向量数据库中的商品名对齐 最终返回确定的商品名列表或者模糊的商品名列表
        Args:
            item_names: LLM提起到商品名列表
        Returns:
            返回确定的容器和options容器
        """
        # 1. 混合检索向量数据库
        search_result: List[Dict[str, Any]] = self._search_vector(item_names)
        if not search_result:
            return [], []
        # 2. 根据混合向量检索结果到结果做对齐confirmed/options
        confirmed, options = self._align(search_result)
        # 3. 分数差异化过滤
        if len(confirmed) > 1:
            confirmed = self._item_name_score_filter(confirmed, search_result)
        # 4. 返回确定的confirmed容器和options容器
        return confirmed, options

    def _item_name_score_filter(self, confirmed: List[str], search_result: List[Dict[str, Any]]):
        """
        判断confirmed中最大得分item与其他item差距多少,如果差距不大保留,过大则直接过滤其他
        Args:
            confirmed: 得分list
            search_result: milvus搜索结果

        Returns:

        """
        #构建名字->最大得分的映射
        item_name_score = {}
        for search_item in search_result:
            search_matchs = search_item.get('item_names', [])
            for item in search_matchs:
                item_score = item.get('score', 0)
                item_name = item.get('item_name', '')
                # 判断item的name在不在confirmed中
                if item_name in confirmed:
                    #如果在,就判断映射里的item_name的值是不是比当前item的score高,高就修改,不高就保留
                    item_name_score[item_name] = max(item_name_score.get(item_name, 0), item_score)
        if not item_name_score:
            return confirmed

        #计算max_score
        max_score = max(item_name_score.values())
        #如果比max_score- score 小太多,就过滤掉
        return [name for name, score in item_name_score.items() if max_score - score <= self._config.item_name_score_gap]




    def _search_vector(self, item_names: List[str]) -> List[Dict[str, Any]]:
        """
        对LLM提取到的所有商品名进行向量化,然后去milvus中检索
        Args:
            item_names: LLM提取到商品名列表

        Returns:
            List[Dict[str, Any]]
            例子：{"extracted_name":"LLM提取的商品名1","matches":[{向量库中查询到的文档1},{向量库中查询到的文档2}]}
            例子：{"extracted_name":"LLM提取的商品名2","matches":[{向量库中查询到的文档1},{向量库中查询到的文档2}]}
        """
        final_search_result = []
        # 1. 获取Milvus客户端
        try:
            milvus_client = StorageClients.get_milvus_client()
        except ConnectionError as e:
            logger.error(f'连接Milvus失败, 原因{str(e)}')
            return final_search_result

        # 2. 获取bge模型
        try:
            bge_model = AIClients.get_bge_m3_client()
        except ConnectionError as e:
            logger.error(f'BGE模型获取失败, 原因{str(e)}')
            return final_search_result

        # 3. 商品名列表向量化(混合向量)
        try:
            hybrid_vector_result = generate_bge_m3_hybrid_vectors(model=bge_model, embedding_documents=item_names)
        except Exception as e:
            logger.error(f'商品列表{item_names}生成混合向量失败 原因:{str(e)}')
            return final_search_result

        # 4. 混合向量检索
        for index, item_name in enumerate(item_names):
            # 构建混合检索请求
            hybrid_requests = create_hybrid_search_requests(hybrid_vector_result['dense'][index],hybrid_vector_result['sparse'][index])
            # 执行混合检索
            hybrid_search_result = execute_hybrid_search_query(milvus_client, self._config.item_name_collection, hybrid_requests, output_fields=['item_name'])
            # 得到解析检索结果
            matches = [{'score': search_item['distance'], 'item_name': search_item['entity']['item_name'] } for search_item in (hybrid_search_result[0] if hybrid_search_result else [])]

            final_search_result.append({
                "extracted_name": item_name,
                "matches": matches
            })
        return final_search_result

    def _align(self, search_result: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
        """
        对齐结果函数
        判断什么向量去confirmed列表什么去options列表
        两个规则：1.如果从向量数据库中查到的商品名分数比如大于0.7 放到confirmed
                2.如果从向量数据库中查到的商品名分数比如小于等于0.7大于0.45 放到options
                3.如果从向量数据库中查到的商品名分数小于等于0.45 两个容器都不放
        confirmed和options中均不应该出现一样的item_name
        Args:
            search_result: 向量数据库检索到的结果
            [
                {
                    "extracted_name": item_name,
                    "matches": matches[{
                        score: xxx
                        item_name
                    }]
                }
            ]
        Returns:
            最终两个容器confirmed、options列表中的商品名
        """
        # 定义两个容器
        confirmed = []
        options = []
        def save_confirmed_item(item):
            if item not in confirmed:
                confirmed.append(item)

        # 遍历检索到的所有商品名的从milvus中的搜索结果
        for search_item in search_result:
            llm_extracted_name = search_item['extracted_name']
            milvus_matches = search_item['matches']
            #milvus_matches对score排序
            sorted_matches = sorted(milvus_matches, key=lambda x: x['score'], reverse=True)

            high_score_items = [item for item in sorted_matches if item['score'] > self._config.item_name_high_confidence]
            #如果存在高分
            if high_score_items:
                # 判断是否有item_name与llm返回的Item相等
                # next函数,第一个参数是一个迭代器对象, next会从这个迭代器中取出下一个元素, 第二个参数是默认值,迭代器耗尽就取它
                extract_item = next((item for item in high_score_items if item['item_name'] == llm_extracted_name), None)
                #有说明匹配到了Llm的item_name直接存进confirmed
                if extract_item:
                    save_confirmed_item(extract_item['item_name'])
                #只有一个也直接存进去
                elif len(high_score_items) == 1:
                    save_confirmed_item(high_score_items[0]['item_name'])
                #多个高分商品名称
                else:
                    #由于分数已经排序过所以取第一个就是最高分
                    top_score = high_score_items[0]['score']
                    #计算第一二个数据的分差,如果差的很大,就只拿第一个(后面的分只会更低
                    if top_score - high_score_items[1]['score'] >= self._config.item_name_score_gap:
                        save_confirmed_item(high_score_items[0]['item_name'])
                    else:
                        #如果差距都不大,就取前item_name_max_options个item放入options,这几个不能在options中,也不能再confirmed
                        for item in high_score_items[:self._config.item_name_max_options]:
                            picked = item['item_name']
                            options_name = [item['item_name'] for item in options]
                            if picked not in confirmed and picked not in options_name:
                                options.append(item)
            #不是高置信度,可能是中置信度
            else:
                mid_score_items = [item for item in sorted_matches if
                                    item['score'] > self._config.item_name_mid_confidence]
                if mid_score_items:
                    #每次选item_name_max_options个item存入options
                    added_count = 0
                    for item in mid_score_items:
                        # 如果已经添加了足够数量的 item，停止循环
                        if added_count >= self._config.item_name_max_options:
                            break
                        options_name = [opt['item_name'] for opt in options]
                        if item['item_name'] not in options_name:
                            options.append(item)
                            added_count += 1

        sored_options = sorted(options, key=lambda x: x['score'], reverse=True)
        options_name = [item['item_name'] for item in sored_options]
        return confirmed, options_name[:self._config.item_name_max_options]

class ItemNameConfirmedNode(BaseNode):
    name = "item_name_confirmed_node"

    def __init__(self):
        super().__init__()
        self._extractor = _ItemNameExtractor()
        self._aligner = _ItemNameAligner()

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        利用LLM从用户原始查询中提取商品名以及改写原始问查询
        llm查到商品然后向量化然后取milvus中查询
        查询到matchs后根据评分得到精确商品名和不精确商品名
        LLM生成答案
        Args:
            state:

        Returns:

        """
        # 1. 获取用户原始问题
        origin_query = state['original_query']
        # 2. 获取历史对话(mongodb)
        history_context = get_recent_messages(session_id=state['session_id'], limit=10)
        formatted_history_str = self._format_history_context(history_context)
        # 3. 利用LLM进行商品名提取和查询重写
        llm_result = self._extractor.extract_item_name(origin_query, formatted_history_str)
        # 3.1 获取LLM结果
        item_names = llm_result.get('item_name')
        rewritten_query = llm_result.get('rewritten_query')
        # 4. 根据item_names做判断
        if item_names:
            confirmed, options = self._aligner.search_and_align(item_names)
        else:
            confirmed, options = [], []

        # 5. 决策
        self._decide(confirmed, options,state,rewritten_query)

        state['history'] = history_context

        return state
    def _format_history_context(self, history_context):
        format_history = []
        for history in history_context:
            role = history['role']
            text = history['text']
            formatted_context = f'角色:{role}, 内容:{text}'
            format_history.append(formatted_context)
        return " ".join(format_history)

    def _decide(self, confirmed: List[str], options: List[str], state: QueryGraphState,
                rewritten_query: str):
        """
        根据confirmed、options来判断是继续检索还是返回用户提示信息
        Args:
            confirmed:  已经确认的商品名列表
            options: 模糊的商品名列表
            state: 查询状态
            rewritten_query: 重写后的问题
            item_names: LLM提取到的商品名列表

        Returns:

        """
        if confirmed:
            state['item_names'] = confirmed #对齐后商品名
            state['rewritten_query'] = rewritten_query
        elif options:
            state["answer"] = (
                f"我不确定您指的是哪款产品。"
                f"您是在询问以下产品吗：{'、'.join(options)}？"
            )
        else:
            state["answer"] = '抱歉，我无法识别您询问的具体产品名称，请提供更准确的产品名称或型号。'


if __name__ == '__main__':
    item_name_confirmed_node = ItemNameConfirmedNode()
    init_state = {
        # "original_query": "RS-12数字万用表和H3C LA2608 室内无线网关的操作区别是什么?"
        # "original_query": "数字万用表如何测量电压?"
        "original_query": "Fluke 数字万用表如何使用呢?"
        # "original_query": "RS-12数字万用表如何测量电压"  # 单个商品询问
    }
    llm_result = item_name_confirmed_node.process(init_state)

    print(llm_result)






