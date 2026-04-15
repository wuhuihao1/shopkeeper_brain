from typing import List, Dict, Any, Tuple
from langchain_openai import ChatOpenAI
from knowledge.processor.query_processor.base import BaseNode, T
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.mongo_history_util import save_chat_message
from knowledge.utils.task_util import set_task_result
from knowledge.utils.sse_util import push_sse_event, SSEEvent
from knowledge.prompts.query_prompt import ANSWER_PROMPT

class AnswerOutPutNode(BaseNode):
    name='answer_output_node'
    def process(self, state: QueryGraphState) -> QueryGraphState:
        """
        核心逻辑:
        1. 从state中获取answer->没有进行三路检索, 不用在生成答案，直接返回.如何推送给前端:1.流式（直接将已经生成的内容都给前端）2.非流式（直接将已经生成的内容都给前端）
        2. 如果没有获取answer->进行了三路检索,需要llm生成答案再返回.如何推送给前端:1.流式（sse）2.非流式（明显变化）
        """
        # 获取数据
        is_stream = state['is_stream']
        task_id = state['task_id']

        #判断是否有答案
        if state.get('answer'):
            #有答案说明三路检索未成功或未确认,直接推给前端
            self._push_exist_answer(task_id, is_stream, state)
            is_streamed = False
        else:
            prompt = self._build_prompt(state)
            state['prompt'] = prompt
            #通过llm生成答案
            self._generate_answer(prompt, task_id, state)
            is_streamed = is_stream
        self.save_history(state)
        if is_stream:
            #已经流式调用把数据传到前端了,就不在给数据
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
                    data={'answer': state.get('answer')}
                )
        return state

    def save_history(self, state: QueryGraphState):
        """
        保存历史对话（Q--->A）
        存储位置：mongodb对应kb001库下的chat_message表中
        """
        try:
            session_id = state['session_id']
            user_query = state['original_query']
            rewritten_query = state['rewritten_query']
            item_names = state['item_names'] or []
            save_chat_message(
                session_id=session_id,
                role='user',
                text=user_query,
                rewritten_query=rewritten_query,
                item_names=item_names,
            )
            save_chat_message(
                session_id=session_id,
                role='assistant',
                text=state.get('answer'),
                rewritten_query=rewritten_query,
                item_names=item_names,
            )
        except Exception as e:
            self.logger.error(f"保存历史对话到MongDB中失败 原因:{str(e)}")


    def _generate_answer(self,prompt: str,task_id: str,state: QueryGraphState):
        """
        调用LLM  生成答案 更新到state
        """
        try:
            client = AIClients.get_llm_client(response_format=False)
        except ConnectionError as e:
            self.logger.error(f'连接llm失败,原因{str(e)}')
            state['answer'] = "LLM暂无法回答"
            return
        if state['is_stream']:
            state['answer'] = self._stream_llm(task_id, prompt, client)
        else:
            state['answer'] = self._invoke_llm(prompt, client)
            # 写入到任务结果队列中(非流式调用)
            set_task_result(task_id=task_id, key="answer", value=state['answer'])

    def _invoke_llm(self,prompt, llm_client):
        """
        非流式调用llm
        """
        try:
            llm_res = llm_client.invoke(prompt)
            if not llm_res:
                return 'LLM暂无法回答'

            llm_content = getattr(llm_res, 'content', "") or ""
            return llm_content
        except Exception as e:
            return 'LLM暂无法回答'

    def _stream_llm(self, task_id, prompt, client):
        """
        流式调用llm
        """
        accelerate_delta = '' #全量数据缓存
        try:
            for chunk in client.stream(prompt):
                delta_text = getattr(chunk, 'content', "")  or ''
                if delta_text:
                    push_sse_event(
                        task_id=task_id,
                        event=SSEEvent.DELTA,
                        data={"delta": delta_text},
                    )
                    accelerate_delta += delta_text
        except Exception as e:
            return "LLM暂无法回答"
        return accelerate_delta

    def _build_prompt(self, state: QueryGraphState) -> str:
        max_context_chars = self.config.max_context_chars
        # 获取必要字段
        user_query = state['rewritten_query']
        item_name = state['item_names'] or []
        # 构建检索上下文, 上下文长度优先给重排序的数据使用
        retrieval_context = state['reranked_docs'] or []
        formatted_context, usage_chars = self._format_retrieval_context(retrieval_context, max_context_chars)
        # 构建历史上下文
        chat_history_context = state.get('history') or [] #从内存获取历史对话
        formatted_history = self._format_chat_history(chat_history_context, usage_chars)

        # 格式化提示词模版
        return ANSWER_PROMPT.format(
            context=formatted_context or "暂无检索到上下文",
            history=formatted_history or "暂无历史上下文",
            item_names=','.join(item_name),
            question=user_query,
        )

    def _format_chat_history(self, chat_history_context: List[Dict[str, Any]], usage_chars: int) -> str:
        """
        格式化历史上下文
        Args:
            chat_history_context: 历史上下文
            usage_chars: 可用字符串长度
        """
        formatted_lines = []
        used_chars = 0
        role_map = {"user": "用户", "assistant": "助手"}
        for msg in chat_history_context:
            role = msg['role']
            text = msg['text']
            if not text or role not in role_map:
                continue
            formatted_line = f"{role_map[role]}: {text}"
            seperator_usage = 1 if formatted_line else 0
            total_length = seperator_usage + len(formatted_line)

            #如果已使用上下文长度+当前行长度大于可用字符,则直接抛弃
            if used_chars + total_length > usage_chars:
                break
            formatted_lines.append(formatted_line)
            used_chars += total_length
        return '\n'.join(formatted_lines)

    def _format_retrieval_context(self, retrieval_context: List[Dict[str, Any]], max_context_chars: int) -> Tuple[str, int]:
        """
        格式化检索到的上下文
        【自己拼接一些元数据：供LLM学习，回答答案更准确】
        Args:
            retrieval_context: 检索到的上下文
            max_context_chars: 最大上下文的长度
        Returns:
            格式后的上下文
        """
        # 1. 遍历
        formatted_lines = []
        usage = 0
        for index, context in enumerate(retrieval_context, 1):
            """
            context格式:
            {
                'content': content,
                'chunk_id': chunk_id,
                'title': title,
                'url': url,
                'source': source,
            }
            """
            content = context.get('content', "")
            if not content:
                continue

            metadata_content = [f'文档{index}']

            #定义元数据模板
            for meta_field, template in [
                                            ('chunk_id', '[chunk_id={}]'),
                                            ('title', '[title={}]'),
                                            ('source', '[source={}]'),
                                            ('url', '[url={}]'),
                                        ]:
                field_value = str(context.get(meta_field, "")).strip()
                if field_value:
                    metadata_content.append(template.format(field_value))
            doc_score = context.get('score')
            if doc_score is not None:
                metadata_content.append(f'[score={doc_score:.6f}]')

            formatted_line = ' '.join(metadata_content) + '\n' + content

            sep_chars = 2 if formatted_lines else 0

            total_length = sep_chars + len(formatted_line)

            #计算当前总长度+之前使用了的长度是否大于最大上下文长度
            if total_length + usage > max_context_chars:
                break
            else:
                formatted_lines.append(formatted_line)
                usage += total_length #检索到的上下文总的长度
        return '\n\n'.join(formatted_lines), max_context_chars - usage

    def _push_exist_answer(self, task_id: str, is_stream: bool, state: QueryGraphState):
        if not is_stream:
            set_task_result(task_id=task_id, key="answer", value=state['answer'])