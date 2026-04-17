"""查询流程主图（听书平台版本）

使用 LangGraph 构建知识库查询工作流。
"""
from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph
from pathlib import Path
from dotenv import load_dotenv
from knowledge.dictionary.query_file import IntentType
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.processor.query_processor.nodes.intent_router_node import IntentRouterNode
from knowledge.processor.query_processor.nodes.book_name_confirmed_node import BookNameConfirmedNode
from knowledge.processor.query_processor.nodes.metadata_filter_node import MetadataFilterNode
from knowledge.processor.query_processor.nodes.hybrid_vector_search_node import HybridVectorSearchNode
from knowledge.processor.query_processor.nodes.hyde_vector_search_node import HyDeVectorSearchNode
from knowledge.processor.query_processor.nodes.web_mcp_search_node import WebMcpSearchNode
from knowledge.processor.query_processor.nodes.rrf_merge_node import RrfMergeNode
from knowledge.processor.query_processor.nodes.reranker_node import RerankerNode
from knowledge.processor.query_processor.nodes.answer_output_node import AnswerOutPutNode

# 显式指定 .env 路径，避免加载到其他项目的配置
_env_path = Path(__file__).resolve().parent.parent.parent / ".env"
load_dotenv(_env_path, override=True)


def route_after_intent(state: QueryGraphState) -> str:
    intent = state.get('intent', IntentType.QA)
    # 直接与枚举比较
    if intent == IntentType.CHAT:
        return "skip_search"

    if intent == IntentType.RECOMMEND:
        return "skip_book_confirm"

    return "need_book_confirm"


def route_after_book_confirm(state: QueryGraphState) -> str:
    """
    书名确认后的路由逻辑
    """
    # 如果有预置答案，直接输出
    if state.get("answer"):
        return "has_answer"

    # 有确认的书名，继续检索
    if state.get("book_names"):
        return "has_book_names"

    # 无结果，直接输出
    return "no_result"


def route_after_merge(state: QueryGraphState) -> str:
    """
    结果融合后的路由逻辑
    """
    rrf_chunks = state.get('rrf_chunks', [])
    if rrf_chunks:
        return "has_results"
    return "no_results"


def create_query_graph() -> CompiledStateGraph:
    """创建查询流程图"""
    workflow = StateGraph(QueryGraphState)

    # 实例化节点
    intent_router_node = IntentRouterNode()
    book_name_confirmed = BookNameConfirmedNode()
    metadata_filter = MetadataFilterNode()
    hybrid_search = HybridVectorSearchNode()
    hyde_search = HyDeVectorSearchNode()
    web_search = WebMcpSearchNode()
    rrf_merge = RrfMergeNode()
    reranker = RerankerNode()
    answer_output = AnswerOutPutNode()

    # 添加节点
    workflow.add_node("intent_router_node", intent_router_node)
    workflow.add_node("book_name_confirmed_node", book_name_confirmed)
    workflow.add_node("metadata_filter_node", metadata_filter)
    workflow.add_node("hybrid_search_node", hybrid_search)
    workflow.add_node("hyde_search_node", hyde_search)
    workflow.add_node("web_search_node", web_search)
    workflow.add_node("join", lambda x: x)
    workflow.add_node("rrf_merge_node", rrf_merge)
    workflow.add_node("reranker_node", reranker)
    workflow.add_node("answer_output_node", answer_output)

    # 设置入口点
    workflow.set_entry_point("intent_router_node")

    # 意图识别后的条件边
    workflow.add_conditional_edges(
        "intent_router_node",
        route_after_intent,
        {
            "skip_search": "answer_output_node",  # 闲聊直接输出
            "skip_book_confirm": "metadata_filter_node",  # 推荐直接走过滤
            "need_book_confirm": "book_name_confirmed_node",  # 其他需要确认书名
        },
    )

    # 书名确认后的条件边
    workflow.add_conditional_edges(
        "book_name_confirmed_node",
        route_after_book_confirm,
        {
            "has_answer": "answer_output_node",
            "has_book_names": "metadata_filter_node",
            "no_result": "answer_output_node",
        },
    )

    # 元数据过滤后，进入多路检索
    workflow.add_edge("metadata_filter_node", "hybrid_search_node")
    workflow.add_edge("metadata_filter_node", "hyde_search_node")
    workflow.add_edge("metadata_filter_node", "web_search_node")

    # 多路检索汇合到join节点
    workflow.add_edge("hybrid_search_node", "join")
    workflow.add_edge("hyde_search_node", "join")
    workflow.add_edge("web_search_node", "join")

    # join后进入RRF融合
    workflow.add_edge("join", "rrf_merge_node")

    # RRF融合后条件边
    workflow.add_conditional_edges(
        "rrf_merge_node",
        route_after_merge,
        {
            "has_results": "reranker_node",
            "no_results": "answer_output_node",
        },
    )

    # 重排序后输出答案
    workflow.add_edge("reranker_node", "answer_output_node")
    workflow.add_edge("answer_output_node", END)

    return workflow.compile()


# 创建全局图实例
query_app = create_query_graph()

if __name__ == "__main__":
    print("=" * 60)
    print("开始测试: 查询流程主图（听书平台版本）")
    print("=" * 60)

    # 测试场景 1：书籍详情查询
    print("\n【场景 1】: 书籍详情查询")
    print("-" * 60)

    mock_state_1 = {
        "original_query": "《活着》这本书讲什么？",
        "session_id": "test_session_001",
        "task_id": "test_task_001",
        "is_stream": False,
    }

    result_1 = query_app.invoke(mock_state_1)

    print(f"\n查询: {mock_state_1['original_query']}")
    print(f"意图: {result_1.get('intent')}")
    print(f"书名: {result_1.get('book_names')}")
    answer_1 = result_1.get("answer", "")
    print(f"答案: {answer_1[:300]}..." if len(answer_1) > 300 else f"答案: {answer_1}")

    # 测试场景 2：书籍推荐
    print("\n\n【场景 2】: 书籍推荐")
    print("-" * 60)

    mock_state_2 = {
        "original_query": "推荐几本好看的科幻小说",
        "session_id": "test_session_002",
        "task_id": "test_task_002",
        "is_stream": False,
    }

    result_2 = query_app.invoke(mock_state_2)

    print(f"\n查询: {mock_state_2['original_query']}")
    print(f"意图: {result_2.get('intent')}")
    answer_2 = result_2.get("answer", "")
    print(f"答案: {answer_2[:300]}..." if len(answer_2) > 300 else f"答案: {answer_2}")

    # 测试场景 3：闲聊
    print("\n\n【场景 3】: 闲聊")
    print("-" * 60)

    mock_state_3 = {
        "original_query": "你好",
        "session_id": "test_session_003",
        "task_id": "test_task_003",
        "is_stream": False,
    }

    result_3 = query_app.invoke(mock_state_3)

    print(f"\n查询: {mock_state_3['original_query']}")
    print(f"意图: {result_3.get('intent')}")
    answer_3 = result_3.get("answer", "")
    print(f"答案: {answer_3}")

    print("\n" + "=" * 60)
    print("全部测试完成")