# RAGAS 评估框架集成设计

## 目标

在现有 LangGraph 查询管道末尾集成 RAGAS 评估，以最小改动实现 RAG 质量评估能力。

## 架构

### 数据流

```
answer_output_node
        │
        ├── (enable_evaluation=False) ──> END
        │
        └── (enable_evaluation=True)  ──> evaluation_node ──> END
```

评估节点位于 `answer_output_node` 之后，通过 `enable_evaluation` 参数控制是否执行。

### 评估指标策略

| 指标 | 始终计算 | 需要 ground_truth | 数据来源 |
|------|---------|------------------|---------|
| Faithfulness | Yes | No | question + answer + contexts |
| Answer Relevancy | Yes | No | question + answer |
| Context Precision | 可选 | Yes | question + contexts + ground_truth |
| Context Recall | 可选 | Yes | question + contexts + ground_truth |

- 始终计算 Faithfulness + Answer Relevancy
- 当请求中提供 `ground_truth` 时，额外计算 Context Precision + Context Recall

### 评估 LLM

复用现有 DashScope API (Qwen-flash)，通过 `langchain_openai.ChatOpenAI` 接入 RAGAS。

## 改动清单

### 1. 新增文件

**`knowledge/processor/query_processor/nodes/evaluation_node.py`**

- 继承 `BaseNode`，遵循现有节点模式
- `process()` 方法：
  1. 从 state 中提取 `original_query`、`answer`、`reranked_docs`（提取 content 作为 contexts）
  2. 构建 RAGAS `SingleTurnSample`
  3. 根据是否有 `ground_truth` 选择指标集合
  4. 调用 `evaluate()` 运行评估
  5. 将评估结果写入 `state['evaluation_result']`
  6. 通过 SSE 推送评估完成事件，非流式写入 task_result
- 使用 `AIClients` 中相同的 API Key/Base URL/Model 配置创建评估用 ChatOpenAI 实例

### 2. 修改文件

**`knowledge/processor/query_processor/state.py`**

`QueryGraphState` 增加 3 个字段：
- `enable_evaluation: bool` — 是否开启评估（默认 False）
- `ground_truth: str` — 可选的标准答案（用于 Context Precision/Recall）
- `evaluation_result: dict` — 评估结果

**`knowledge/processor/query_processor/main_graph.py`**

- 导入 `EvaluationNode`
- 添加条件路由函数 `route_after_answer`：当 `enable_evaluation=True` 时路由到 `evaluation_node`，否则到 `END`
- 将 `answer_output_node -> END` 改为 `answer_output_node -> 条件路由`

**`knowledge/schema/query_schema.py`**

`QueryRequest` 增加 2 个字段：
- `enable_evaluation: bool = False`
- `ground_truth: Optional[str] = None`

**`knowledge/service/query_service.py`**

`run_query_graph()` 方法签名增加 `enable_evaluation` 和 `ground_truth` 参数，透传到初始 state。

**`knowledge/api/query_router.py`**

`query_endpoint` 中将新参数从 request 透传到 service 调用。

**`knowledge/requirements.txt`**

添加 `ragas` 依赖。

## RAGAS 配置

```python
# 复用 DashScope API，通过 langchain_openai.ChatOpenAI 接入
from langchain_openai import ChatOpenAI

evaluator_llm = ChatOpenAI(
    model_name="qwen-flash",
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    openai_api_base=os.getenv("OPENAI_API_BASE"),
    temperature=0,
)
```

## 评估结果格式

```python
{
    "faithfulness": 0.85,
    "answer_relevancy": 0.92,
    "context_precision": 0.78,   # 仅在提供 ground_truth 时存在
    "context_recall": 0.65,      # 仅在提供 ground_truth 时存在
}
```

## 设计原则

- **最小改动**：只增加 1 个新文件，修改 6 个文件，每处修改仅几行
- **遵循现有模式**：新节点继承 BaseNode，与现有节点风格一致
- **完全可选**：默认不开启，不影响现有功能
- **代码精简带注释**：关键逻辑加中文注释
