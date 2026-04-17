"""RAGAS 评估节点

在查询管道末尾运行 RAGAS 评估，评估 RAG 系统的检索和生成质量。
通过 enable_evaluation 参数控制是否启用。
"""
import os, json
import asyncio as _asyncio
import math

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas import evaluate
from ragas.dataset_schema import SingleTurnSample, EvaluationDataset
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall

from knowledge.processor.query_processor.base import BaseNode
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.utils.task_util import set_task_result
from knowledge.utils.sse_util import push_sse_event, SSEEvent


class EvaluationNode(BaseNode):
    """RAGAS 评估节点 — 评估检索和生成质量，结果写入 state['evaluation_result']"""
    name = 'evaluation_node'

    def process(self, state: QueryGraphState) -> QueryGraphState:
        # ---------- 1. 从 state 中提取评估所需数据 ----------
        question = state.get('original_query', '')
        answer = state.get('answer', '')
        # 从重排序结果中提取上下文文本列表
        reranked_docs = state.get('reranked_docs', [])
        ground_truth = state.get('ground_truth', '')
        is_stream = state.get('is_stream', False)

        # 基础校验：缺少问题或答案则跳过评估
        if not question or not answer:
            self.logger.warning('缺少问题或答案，跳过评估')
            state['evaluation_result'] = {'error': '缺少问题或答案'}
            return state

        # 流式模式下，检查答案是否完整（至少10个字符）
        if is_stream and len(answer.strip()) < 10:
            self.logger.warning(f'流式模式下答案不完整或太短 (长度: {len(answer)}), 跳过评估')
            state['evaluation_result'] = {'error': '流式答案不完整', 'status': 'skipped'}
            return state

        # ---------- 2. 清洗和验证 contexts ----------
        # 确保问题、答案是纯字符串（防止 list/dict 等类型混入）
        if not isinstance(question, str):
            question = str(question) if question is not None else ''
        if not isinstance(answer, str):
            answer = str(answer) if answer is not None else ''

        clean_contexts = []
        for doc in reranked_docs:
            # doc 可能是 dict 或其他对象
            if isinstance(doc, dict):
                content = doc.get('content', '')
            elif isinstance(doc, str):
                content = doc
            else:
                content = getattr(doc, 'content', '') or getattr(doc, 'page_content', '') or ''
            # 确保每个 context 元素是 str
            if content and isinstance(content, str):
                clean_contexts.append(content)
            elif content is not None:
                self.logger.debug(f'转换非字符串上下文: {type(content)}')
                clean_contexts.append(str(content))
            else:
                self.logger.debug(f'跳过空上下文')

        # Ragas 要求 contexts 不能为空，也不能包含非字符串
        if not clean_contexts:
            self.logger.warning('没有有效的上下文，使用空字符串占位')
            clean_contexts = [""]

        # 确保 ground_truth 是字符串或 None
        ground_truth_str = ground_truth if isinstance(ground_truth, str) and ground_truth else None

        # ---------- 3. 构建 RAGAS 评估样本 ----------
        try:
            # 使用字典方式构建（更稳定，避免字段名问题）
            from datasets import Dataset

            data = {
                "user_input": [question],
                "response": [answer],
                "retrieved_contexts": [clean_contexts],
                "reference": [ground_truth_str] if ground_truth_str else [None],
            }
            dataset = Dataset.from_dict(data)
            self.logger.info(f'使用 Dataset 方式构建数据集成功')

        except Exception as e:
            self.logger.warning(f'Dataset 方式构建失败: {e}，回退到 SingleTurnSample')
            # 回退到 SingleTurnSample 方式
            sample = SingleTurnSample(
                user_input=question,
                response=answer,
                retrieved_contexts=clean_contexts,
                reference=ground_truth_str,
            )
            dataset = EvaluationDataset(samples=[sample])

        # ---------- 4. 选择评估指标 ----------
        metrics = [Faithfulness(), AnswerRelevancy()]

        # 当提供标准答案时，额外计算 ContextPrecision 和 ContextRecall
        if ground_truth_str:
            self.logger.info('检测到标准答案，额外计算 ContextPrecision 和 ContextRecall')
            metrics.extend([ContextPrecision(), ContextRecall()])

        # 记录使用的指标
        metric_names = [getattr(m, 'name', str(m)) for m in metrics]
        self.logger.info(f'使用的评估指标: {metric_names}')

        # ---------- 5. 创建评估用 LLM 和 Embeddings ----------
        evaluator_llm = ChatOpenAI(
            model_name=os.getenv('LLM_DEFAULT_MODEL', 'qwen-flash'),
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            openai_api_base=os.getenv('OPENAI_API_BASE'),
            temperature=0,
        )

        # AnswerRelevancy 需要 Embeddings 来计算语义相似度
        evaluator_embeddings = OpenAIEmbeddings(
            model=os.getenv('EMBEDDING_MODEL', 'text-embedding-v3'),
            openai_api_key=os.getenv('OPENAI_API_KEY'),
            openai_api_base=os.getenv('OPENAI_API_BASE'),
        )

        # ---------- 6. 运行评估（兼容 RAGAS 同步/异步 API） ----------
        try:
            result = self._run_evaluate(dataset, metrics, evaluator_llm, evaluator_embeddings)
        except Exception as e:
            self.logger.error(f'RAGAS 评估失败: {e}')
            state['evaluation_result'] = {'error': str(e)}
            return state

        # ---------- 7. 提取评估分数 ----------
        eval_result = self._extract_scores(result)

        # 处理 nan 值（来自 AnswerRelevancy 等指标）
        eval_result = self._clean_nan_values(eval_result)

        state['evaluation_result'] = eval_result
        self.logger.info(f'RAGAS 评估结果: {eval_result}')

        # ---------- 8. 推送评估结果 ----------
        task_id = state.get('task_id', '')
        if is_stream and task_id:
            # 流式模式：通过 SSE 推送 EVALUATION 事件
            push_sse_event(task_id=task_id, event=SSEEvent.FINAL, data={'evaluation': eval_result})
        elif task_id:
            # 非流式模式：写入任务结果队列
            set_task_result(task_id=task_id, key='evaluation', value=json.dumps(eval_result))

        return state

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def _run_evaluate(self, dataset, metrics, llm, embeddings):
        """运行 RAGAS evaluate，兼容同步和异步版本。"""
        # 如果 embeddings 为 None，只传 llm
        if embeddings is None:
            coro = evaluate(dataset=dataset, metrics=metrics, llm=llm)
        else:
            coro = evaluate(dataset=dataset, metrics=metrics, llm=llm, embeddings=embeddings)

        if _asyncio.iscoroutine(coro):
            try:
                # 检查是否已有运行中的事件循环
                _ = _asyncio.get_running_loop()
                # 有运行中的循环，使用 nest_asyncio
                try:
                    import nest_asyncio
                    nest_asyncio.apply()
                except ImportError:
                    self.logger.warning('nest_asyncio 未安装，可能导致事件循环冲突')
                return _asyncio.run(coro)
            except RuntimeError:
                # 没有运行中的事件循环，直接 run
                return _asyncio.run(coro)
        return coro

    @staticmethod
    def _extract_scores(result) -> dict:
        """从 RAGAS EvaluationResult 中提取分数。"""
        # RAGAS 0.2.x: result.scores 是一个 list[dict]，每个元素对应一个样本
        if hasattr(result, 'scores') and result.scores:
            raw = result.scores[0] if result.scores else {}
        # 兼容新版 RAGAS 0.3.x: result 可能是字典
        elif isinstance(result, dict):
            raw = result
        # 兼容 result 有 __dict__ 属性
        elif hasattr(result, '__dict__'):
            raw = result.__dict__
        else:
            return {'error': '无法解析评估结果'}

        # 提取分数，保留 4 位小数
        scores = {}
        for key, value in raw.items():
            # 跳过非数值字段
            if key.startswith('_'):
                continue
            if isinstance(value, (int, float)):
                scores[key] = round(value, 4)
            elif value is not None:
                scores[key] = value
        return scores

    @staticmethod
    def _clean_nan_values(scores: dict) -> dict:
        """将字典中的 nan 值替换为 None"""
        cleaned = {}
        for key, value in scores.items():
            if isinstance(value, float) and math.isnan(value):
                cleaned[key] = None
            else:
                cleaned[key] = value
        return cleaned