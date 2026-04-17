import json
import re
from typing import Dict, Any, Optional

from langchain_core.messages import SystemMessage, HumanMessage
from knowledge.dictionary.query_file import IntentType
from knowledge.processor.query_processor.base import BaseNode
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.processor.query_processor.exceptions import StateFieldError
from knowledge.utils.client.ai_clients import AIClients
from knowledge.prompts.query_prompt import INTENT_ROUTER_USER_PROMPT_TEMPLATE, INTENT_ROUTER_SYSTEM_PROMPT



class IntentRouterNode(BaseNode):
    """
    意图识别节点
    只负责识别意图类型，不提取书名（书名由 BookNameConfirmedNode 负责）
    """
    name = "intent_router_node"

    def process(self, state: QueryGraphState) -> QueryGraphState:
        """核心逻辑：识别用户意图"""
        # 1. 获取用户问题
        original_query = state.get('original_query', '')
        if not original_query:
            raise StateFieldError(
                node_name=self.name,
                field_name='original_query',
                expected_type=str,
                message='用户问题不能为空'
            )

        # 2. 获取历史对话上下文
        history_context = self._format_history_context(state.get('history', []))

        # 3. 调用LLM识别意图
        intent_result = self._recognize_intent(original_query, history_context)

        # 4. 将意图存入state（不存书名）
        state['intent'] = intent_result.get('intent', IntentType.QA)
        state['intent_confidence'] = intent_result.get('confidence', 0.5)
        state['intent_reason'] = intent_result.get('reason', '')

        self.logger.info(
            f"意图识别结果 - 查询: {original_query[:50]}... "
            f"意图: {state['intent']} "
            f"置信度: {state['intent_confidence']}"
        )

        if not state.get('rewritten_query'):
            state['rewritten_query'] = original_query
        return state

    def _format_history_context(self, history: list) -> str:
        """格式化历史对话上下文"""
        if not history:
            return "无历史对话"

        formatted_lines = []
        role_map = {"user": "用户", "assistant": "助手"}

        # 只取最近5条对话作为上下文
        recent_history = history[-5:] if len(history) > 5 else history

        for msg in recent_history:
            role = msg.get('role', '')
            text = msg.get('text', '')
            if role in role_map and text:
                formatted_lines.append(f"{role_map[role]}: {text}")

        return '\n'.join(formatted_lines) if formatted_lines else "无历史对话"

    def _recognize_intent(self, query: str, history_context: str) -> Dict[str, Any]:
        """
        调用LLM识别意图
        Args:
            query: 用户问题
            history_context: 历史对话上下文
        Returns:
            意图识别结果字典
        """
        # 1. 先尝试规则匹配（快速路径）
        rule_result = self._rule_based_intent(query)
        if rule_result and rule_result.get('confidence', 0) > 0.8:
            self.logger.info(f"使用规则匹配识别意图: {rule_result}")
            return rule_result

        # 2. 调用LLM识别
        try:
            llm_client = AIClients.get_llm_client(response_format=True)
        except ConnectionError as e:
            self.logger.error(f"LLM客户端获取失败: {str(e)}，使用规则匹配兜底")
            return self._rule_based_intent(query) or self._default_intent()

        user_prompt = INTENT_ROUTER_USER_PROMPT_TEMPLATE.format(
            history_context=history_context,
            query=query
        )

        try:
            llm_response = llm_client.invoke([
                SystemMessage(content=INTENT_ROUTER_SYSTEM_PROMPT),
                HumanMessage(content=user_prompt)
            ])

            content = llm_response.content.strip()
            result = self._parse_llm_response(content)
            return result

        except Exception as e:
            self.logger.error(f"LLM意图识别失败: {str(e)}，使用规则匹配兜底")
            return self._rule_based_intent(query) or self._default_intent()

    def _parse_llm_response(self, content: str) -> Dict[str, Any]:
        """解析LLM返回的JSON响应"""
        try:
            content = re.sub(r'^```(?:json)?\s*', '', content.strip())
            content = re.sub(r'\s*```$', '', content)
            result = json.loads(content)

            intent = result.get('intent', 'qa')
            if intent not in [e.value for e in IntentType]:
                intent = 'qa'

            return {
                'intent': intent,
                'confidence': float(result.get('confidence', 0.5)),
                'reason': result.get('reason', '')
            }
        except json.JSONDecodeError:
            self.logger.warning(f"JSON解析失败，原始内容: {content}")
            return self._default_intent()

    def _rule_based_intent(self, query: str) -> Optional[Dict[str, Any]]:
        """基于规则的快速意图识别"""
        if not query:
            return None

        # 详情意图关键词（高优先级）
        detail_keywords = ['讲什么', '介绍', '作者', '是谁', '时长', '演播', '标签', '亮点', '内容']
        for kw in detail_keywords:
            if kw in query:
                return {
                    'intent': IntentType.DETAIL,
                    'confidence': 0.85,
                    'reason': f'包含详情关键词: {kw}'
                }

        # 推荐意图关键词
        recommend_keywords = ['推荐', '有什么', '哪本', '适合', '好看', '经典', '必听', '求推荐']
        for kw in recommend_keywords:
            if kw in query:
                return {
                    'intent': IntentType.RECOMMEND,
                    'confidence': 0.85,
                    'reason': f'包含推荐关键词: {kw}'
                }

        # 检索意图关键词
        search_keywords = ['查询', '找', '笔记', '评论', '摘要', '搜索']
        for kw in search_keywords:
            if kw in query:
                return {
                    'intent': IntentType.SEARCH,
                    'confidence': 0.8,
                    'reason': f'包含检索关键词: {kw}'
                }

        # 问答意图关键词
        qa_keywords = ['什么是', '为什么', '怎么理解', '区别', '解释', '意义']
        for kw in qa_keywords:
            if kw in query:
                return {
                    'intent': IntentType.QA,
                    'confidence': 0.8,
                    'reason': f'包含问答关键词: {kw}'
                }

        # 闲聊意图
        chat_keywords = ['你好', '谢谢', '再见', '感谢', '帮忙']
        for kw in chat_keywords:
            if kw in query:
                return {
                    'intent': IntentType.CHAT,
                    'confidence': 0.9,
                    'reason': f'包含闲聊关键词: {kw}'
                }

        # 检测到书名号但没有其他关键词，默认详情
        if re.search(r'《[^》]+》', query):
            return {
                'intent': IntentType.DETAIL,
                'confidence': 0.7,
                'reason': '检测到书名号，默认为详情意图'
            }

        return None

    def _default_intent(self) -> Dict[str, Any]:
        """默认意图（兜底）"""
        return {
            'intent': IntentType.QA,
            'confidence': 0.5,
            'reason': '无法识别意图，默认使用问答'
        }


if __name__ == '__main__':
    # 测试代码
    class MockConfig:
        pass


    test_queries = [
        "推荐几本好看的科幻小说",
        "《活着》这本书讲什么",
        "查询红楼梦的听书笔记",
        "什么是生命韧性",
        "你好",
        "通勤适合听什么书",
        "余华是谁",
    ]

    node = IntentRouterNode()
    node.config = MockConfig()

    print("=" * 60)
    print("意图识别测试")
    print("=" * 60)

    for query in test_queries:
        state = {
            "original_query": query,
            "history": []
        }
        result = node.process(state)
        print(f"\n问题: {query}")
        print(f"意图: {result.get('intent')}")
        print(f"置信度: {result.get('intent_confidence')}")
        print("-" * 40)