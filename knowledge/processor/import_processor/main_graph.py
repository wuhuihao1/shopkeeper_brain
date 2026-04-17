import json

from langgraph.graph.state import StateGraph, CompiledStateGraph
from langgraph.graph import END

from knowledge.processor.import_processor.base import setup_logging
from knowledge.processor.import_processor.nodes.pdf_to_md_node import PdfToMdNode
from knowledge.processor.import_processor.nodes.entry_node import EntryNode
from knowledge.processor.import_processor.nodes.document_split_node import DocumentSplitNode
from knowledge.processor.import_processor.nodes.embedding_chunks_node import EmbeddingChunksNode
from knowledge.processor.import_processor.nodes.import_milvus_node import ImportMilvusNode
from knowledge.processor.import_processor.nodes.book_name_recognition_node import BookNameRecognitionNode
from knowledge.processor.import_processor.state import ImportGraphState

"""
编排节点

定义节点
定义条件边
定义顺序边
运行整个pineline图谱的各个节点

"""
def import_router(state: ImportGraphState):
    """
    根据状态决定到达的下一个节点
    Args:
        state:

    Returns:

    """
    if state.get('is_pdf_read_enabled'):
        return 'pdf_to_md_node'
    elif state.get('is_md_read_enabled'):
        return 'document_split_node'
    else:
        return END

def import_graph() -> CompiledStateGraph:
    """
    主函数
    定义运行时图状态workflow
    定义节点
    定义边
    返回各个节点运行状态
    Returns:
        states
    """
    # 1. 定义运行时图状态workflow
    workflow = StateGraph(ImportGraphState)
    # 2. 定义入口节点
    workflow.set_entry_point('entry_node')
    # 3. 定义其它节点名和节点实例的映射表
    node_name_obj={
        'pdf_to_md_node': PdfToMdNode(),
        'entry_node': EntryNode(),
        'document_split_node': DocumentSplitNode(),
        'item_name_recognition_node': BookNameRecognitionNode(),
        'embedding_chunks_node': EmbeddingChunksNode(),
        "import_milvus_node": ImportMilvusNode(),
    }
    # 4. 遍历映射表添加
    for node_name, node_object in node_name_obj.items():
        workflow.add_node(node_name, node_object)
    # 5. 定义条件边
    #左边的是import_router返回值,右边的是具体的节点名
    workflow.add_conditional_edges('entry_node', import_router, {
        'document_split_node': 'document_split_node',
        'pdf_to_md_node': 'pdf_to_md_node',
        END: END
    })
    # 5.2 定义业务边
    workflow.add_edge('document_split_node', 'item_name_recognition_node')
    workflow.add_edge('item_name_recognition_node', 'embedding_chunks_node')
    workflow.add_edge('embedding_chunks_node', 'import_milvus_node')
    workflow.add_edge('import_milvus_node', END)


    # 5.3 编译
    complied_state_graph = workflow.compile()
    # 5.4 返回
    return complied_state_graph

import_app = import_graph()


def run_import_graph():
    # 1. 定义运行graph流程的状态

    graph_state = {
        "import_file_path": r"W:\test\PythonProject\smart_audiobook\docs\活着简介.md",
        "file_dir": r"W:\test\PythonProject\shopkeeper_brain\knowledge\processor\import_processor\temp_dir"

    }

    # stream:迭代整个graph图状态可以得到每一个节点的事件(节点的名字以及节点操作完state之后的新状态)

    for event in import_app.stream(graph_state):
        final_state = {}
        for key, value in event.items():
            print(f"当前正在执行的节点：{key}")
            print(f"当前正在执行的节点状态:{value}")
            final_state = value
    return final_state

if __name__ == '__main__':
    setup_logging()
    final_state = run_import_graph()

    # 整个执行的状态图(方便观察) ascii
    print(import_app.get_graph().print_ascii())