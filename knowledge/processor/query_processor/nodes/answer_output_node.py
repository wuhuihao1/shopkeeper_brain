from typing import List, Dict, Any, Tuple
from langchain_openai import ChatOpenAI
from knowledge.processor.query_processor.base import BaseNode
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.mongo_history_util import save_chat_message
from knowledge.utils.task_util import set_task_result
from knowledge.utils.sse_util import push_sse_event, SSEEvent
from knowledge.prompts.query_prompt import ANSWER_PROMPT_BOOK


class AnswerOutPutNode(BaseNode):
    """
    答案输出节点（听书平台版本）
    1. 如果有预置答案（如书名无法确认），直接返回
    2. 否则基于重排序后的文档生成答案
    3. 支持流式和非流式输出
    4. 保存对话历史到MongoDB
    """
    name = 'answer_output_node'

    def process(self, state: QueryGraphState) -> QueryGraphState:
        # 获取配置
        is_stream = state.get('is_stream', False)
        task_id = state.get('task_id', '')

        # 判断是否有预置答案
        if state.get('answer'):
            # 有预置答案（如书名无法确认），直接返回
            self._push_existing_answer(task_id, is_stream, state)
            is_streamed = False
        else:
            # 构建Prompt并生成答案
            prompt = self._build_prompt(state)
            state['prompt'] = prompt
            self._generate_answer(prompt, task_id, state, is_stream)
            is_streamed = is_stream

        # 保存对话历史
        self._save_history(state)

        # 推送最终完成事件
        if is_stream:
            if is_streamed:
                push_sse_event(
                    task_id=task_id,
                    event=SSEEvent.FINAL,
                    data={}
                )
            else:
                push_sse_event(
                    task_id=task_id,
                    event=SSEEvent.FINAL,
                    data={'answer': state.get('answer', '')}
                )

        return state

    def _save_history(self, state: QueryGraphState) -> None:
        """
        保存历史对话到MongoDB
        """
        try:
            session_id = state.get('session_id', '')
            original_query = state.get('original_query', '')
            rewritten_query = state.get('rewritten_query', '')
            book_names = state.get('book_names', []) or []
            answer = state.get('answer', '')

            if not session_id:
                self.logger.warning("session_id为空，跳过保存历史")
                return

            # 保存用户问题
            save_chat_message(
                session_id=session_id,
                role='user',
                text=original_query,
                rewritten_query=rewritten_query,
                book_names=book_names,
            )

            # 保存助手回答
            save_chat_message(
                session_id=session_id,
                role='assistant',
                text=answer,
                rewritten_query=rewritten_query,
                book_names=book_names,
            )
            self.logger.info(f"对话历史保存成功，session_id: {session_id}")
        except Exception as e:
            self.logger.error(f"保存历史对话到MongoDB失败: {str(e)}")

    def _generate_answer(self, prompt: str, task_id: str, state: QueryGraphState, is_stream: bool) -> None:
        """
        调用LLM生成答案
        Args:
            prompt: 提示词
            task_id: 任务ID
            state: 状态对象
            is_stream: 是否流式输出
        """
        try:
            llm_client = AIClients.get_llm_client(response_format=False)
        except ConnectionError as e:
            self.logger.error(f'连接LLM失败: {str(e)}')
            state['answer'] = "抱歉，LLM服务暂时不可用，请稍后重试。"
            return

        if is_stream:
            state['answer'] = self._stream_llm(task_id, prompt, llm_client)
        else:
            state['answer'] = self._invoke_llm(prompt, llm_client)
            # 非流式调用，写入任务结果队列
            set_task_result(task_id=task_id, key="answer", value=state['answer'])

    def _invoke_llm(self, prompt: str, llm_client: ChatOpenAI) -> str:
        """
        非流式调用LLM
        Args:
            prompt: 提示词
            llm_client: LLM客户端
        Returns:
            生成的答案
        """
        try:
            llm_response = llm_client.invoke(prompt)
            if not llm_response:
                return "抱歉，无法生成回答，请稍后重试。"

            content = getattr(llm_response, 'content', '')
            return content if content else "抱歉，无法生成回答，请稍后重试。"
        except Exception as e:
            self.logger.error(f"LLM调用失败: {str(e)}")
            return "抱歉，LLM服务暂时不可用，请稍后重试。"

    def _stream_llm(self, task_id, prompt, client):
        """
        流式调用LLM
        """
        accumulate_delta = ''
        try:
            for chunk in client.stream(prompt):
                delta_text = getattr(chunk, 'content', "") or ''
                if delta_text:
                    push_sse_event(
                        task_id=task_id,
                        event=SSEEvent.DELTA,
                        data={"delta": delta_text},
                    )
                    accumulate_delta += delta_text

            # 流式完成后，发送 FINAL 事件
            push_sse_event(
                task_id=task_id,
                event=SSEEvent.FINAL,
                data={"answer": accumulate_delta}
            )

        except Exception as e:
            self.logger.error(f"流式LLM调用失败: {str(e)}")
            push_sse_event(
                task_id=task_id,
                event=SSEEvent.FINAL,
                data={"error": str(e)}
            )
            return "抱歉，LLM服务暂时不可用，请稍后重试。"

        return accumulate_delta

    def _build_prompt(self, state: QueryGraphState) -> str:
        """
        构建LLM提示词
        Args:
            state: 状态对象
        Returns:
            格式化的提示词
        """
        max_context_chars = getattr(self.config, 'max_context_chars', 4000)

        # 获取必要字段
        user_query = state.get('rewritten_query', '') or state.get('original_query', '')
        book_names = state.get('book_names', []) or []
        intent = state.get('intent', 'qa')

        # 构建检索上下文
        retrieval_context = state.get('reranked_docs', []) or []
        formatted_context, remaining_chars = self._format_retrieval_context(
            retrieval_context, max_context_chars
        )

        # 构建历史对话上下文
        chat_history = state.get('history', []) or []
        formatted_history = self._format_chat_history(chat_history, remaining_chars)

        # 根据意图调整Prompt风格
        intent_hint = self._get_intent_hint(intent)

        # 格式化提示词
        return ANSWER_PROMPT_BOOK.format(
            intent_hint=intent_hint,
            context=formatted_context or "暂无检索到相关内容",
            history=formatted_history or "暂无历史对话",
            book_names='、'.join(book_names) if book_names else "未指定",
            question=user_query,
        )

    def _get_intent_hint(self, intent: str) -> str:
        """
        根据意图类型返回提示
        Args:
            intent: 意图类型
        Returns:
            意图提示字符串
        """
        hints = {
            'recommend': '请根据检索到的内容，向用户推荐合适的书籍，说明推荐理由和适合人群。',
            'detail': '请根据检索到的内容，详细介绍这本书的相关信息，包括内容简介、作者介绍等。',
            'search': '请根据检索到的内容，返回用户查询的具体信息，并注明来源。',
            'qa': '请根据检索到的内容，回答用户的问题。如果检索内容不足以回答，请如实告知。',
            'chat': '请友好地回应用户的问候或感谢。',
        }
        return hints.get(intent, hints['qa'])

    def _format_chat_history(self, chat_history: List[Dict[str, Any]], max_chars: int) -> str:
        """
        格式化历史对话上下文
        Args:
            chat_history: 历史对话列表
            max_chars: 最大字符数
        Returns:
            格式化后的历史对话字符串
        """
        if not chat_history:
            return ""

        formatted_lines = []
        used_chars = 0
        role_map = {"user": "用户", "assistant": "助手"}

        # 从最新的对话开始，但保持顺序
        for msg in chat_history:
            role = msg.get('role', '')
            text = msg.get('text', '')
            if not text or role not in role_map:
                continue

            formatted_line = f"{role_map[role]}: {text}"
            total_length = len(formatted_line) + 1  # +1 for newline

            if used_chars + total_length > max_chars:
                break

            formatted_lines.append(formatted_line)
            used_chars += total_length

        return '\n'.join(formatted_lines)

    def _format_retrieval_context(self, retrieval_context: List[Dict[str, Any]],
                                   max_chars: int) -> Tuple[str, int]:
        """
        格式化检索到的上下文
        Args:
            retrieval_context: 检索到的文档列表
            max_chars: 最大字符数
        Returns:
            (格式化后的上下文, 剩余可用字符数)
        """
        if not retrieval_context:
            return "", max_chars

        formatted_lines = []
        used_chars = 0

        for idx, doc in enumerate(retrieval_context, 1):
            content = doc.get('content', '')
            if not content:
                continue

            # 构建元数据字符串
            metadata_parts = [f'[文档{idx}]']

            book_name = doc.get('book_name', '')
            if book_name:
                metadata_parts.append(f'[书名: {book_name}]')

            content_type = doc.get('content_type', '')
            if content_type:
                metadata_parts.append(f'[类型: {content_type}]')

            author_name = doc.get('author_name', '')
            if author_name:
                metadata_parts.append(f'[作者: {author_name}]')

            source = doc.get('source', '')
            if source == 'web':
                url = doc.get('url', '')
                if url:
                    metadata_parts.append(f'[来源: {url}]')
            else:
                source_file = doc.get('source_file', '')
                if source_file:
                    metadata_parts.append(f'[来源: {source_file}]')

            score = doc.get('score')
            if score is not None:
                metadata_parts.append(f'[相关度: {score:.3f}]')

            metadata_str = ' '.join(metadata_parts)
            formatted_line = f"{metadata_str}\n{content}"

            total_length = len(formatted_line) + 2  # +2 for double newline

            if used_chars + total_length > max_chars:
                break

            formatted_lines.append(formatted_line)
            used_chars += total_length

        result = '\n\n'.join(formatted_lines)
        remaining_chars = max_chars - used_chars

        return result, remaining_chars

    def _push_existing_answer(self, task_id: str, is_stream: bool, state: QueryGraphState) -> None:
        """
        推送已有的答案（非LLM生成的预置答案）
        Args:
            task_id: 任务ID
            is_stream: 是否流式
            state: 状态对象
        """
        if not is_stream:
            set_task_result(task_id=task_id, key="answer", value=state.get('answer', ''))


if __name__ == '__main__':
    # 测试代码（config 由 BaseNode 自动提供）
    node = AnswerOutPutNode()

    # 模拟测试数据
    mock_state = {
        "original_query": "《活着》讲什么",
        "rewritten_query": "《活着》这本书主要讲了什么内容？",
        "book_names": ["活着"],
        "intent": "detail",
        "is_stream": False,
        "task_id": "test_001",
        "session_id": "session_001",
        "reranked_docs": [
            {
                "content": "《活着》通过主人公福贵一生的遭遇，书写普通人如何在一次次失去中继续活下去。",
                "book_name": "活着",
                "content_type": "书籍简介",
                "author_name": "余华",
                "source": "local",
                "source_file": "活着_简介.md",
                "score": 0.85
            },
            {
                "content": "余华是中国当代重要作家之一，其作品语言简洁但情感力量很强。",
                "book_name": "活着",
                "content_type": "作者介绍",
                "author_name": "余华",
                "source": "local",
                "source_file": "活着_作者介绍.md",
                "score": 0.75
            }
        ],
        "history": []
    }

    result = node.process(mock_state)

