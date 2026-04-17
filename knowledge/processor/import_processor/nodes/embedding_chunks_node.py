"""
BGE-M3 Embedding Chunks Node
负责将文档chunks进行批量向量嵌入（稠密+稀疏），并将结果注入到每个chunk中。
"""
from typing import List, Dict, Any
from pathlib import Path
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from knowledge.processor.import_processor.base import BaseNode, setup_logging
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.processor.import_processor.exceptions import ValidationError, EmbeddingError
from knowledge.utils.client.ai_clients import AIClients


class EmbeddingChunksNode(BaseNode):
    """
    对所有 chunks 进行 BGE-M3 向量嵌入：
    1. 校验输入，获取 chunks 列表
    2. 按批次拼接 embedding_content（book_name + content），调用模型批量编码
    3. 从编码结果中提取稠密向量和稀疏向量，注入到每个 chunk 中
    """
    name = 'embedding_chunks_node'

    def process(self, state: ImportGraphState) -> ImportGraphState:
        # 1. 参数校验
        self.log_step("step1", "校验chunks的数据结构")
        chunks = self._validate_inputs(state)

        # 2. 获取批量嵌入的阈值，默认配置16
        self.log_step("step2", "获取BGE-M3嵌入模型客户端")
        try:
            embed_model = AIClients.get_bge_m3_client()
        except ConnectionError as e:
            self.logger.error(f"BGE-M3嵌入模型创建失败,原因:{str(e)}")
            raise EmbeddingError(message=f"BGE-M3嵌入模型创建失败,原因:{str(e)}", node_name=self.name)

        # 3. 批量嵌入
        embedding_batch_size = getattr(self.config, 'embedding_batch_size', 16)
        total = len(chunks)

        final_chunks = []
        for index in range(0, total, embedding_batch_size):
            batch_end = min(index + embedding_batch_size, total)
            batch_chunk = chunks[index:batch_end]
            self.logger.info(f'嵌入批次 [{index + 1}-{batch_end}] / {total}')
            current_chunks = self._embed_chunks(batch_chunk, embed_model)
            final_chunks.extend(current_chunks)

        # 4. 更新state的chunks
        state['chunks'] = final_chunks
        return state

    def _embed_chunks(self, batch_chunks: List[Dict[str, Any]], embed_model: BGEM3EmbeddingFunction) -> List[
        Dict[str, Any]]:
        """
        批量嵌入chunks
        Args:
            batch_chunks: 批量chunks
            embed_model: 嵌入式向量模型
        Returns:
            返回得到向量化数据的chunks
        """
        # 使用 book_name（新项目）而非 item_name（旧项目）
        embedding_documents = [
            f"{chunk.get('book_name', '')}\n{chunk.get('content', '')}"
            for chunk in batch_chunks
        ]

        # 嵌入chunks的真正内容
        try:
            embed_vector = embed_model.encode_documents(embedding_documents)
        except Exception as e:
            raise EmbeddingError(message=f"嵌入失败,原因:{str(e)}", node_name=self.name)

        if not embed_vector:
            raise EmbeddingError(message='嵌入结果不存在', node_name=self.name)

        # 获取稀疏压缩矩阵
        sparse_csr = embed_vector['sparse']

        for index, chunk in enumerate(batch_chunks):
            chunk['dense_vector'] = embed_vector['dense'][index].tolist()
            chunk['sparse_vector'] = self._extract_sparse_vector(index, sparse_csr)

        return batch_chunks

    def _extract_sparse_vector(self, index: int, sparse_csr):
        """从CSR矩阵中提取稀疏向量"""
        start_index = sparse_csr.indptr[index]
        end_index = sparse_csr.indptr[index + 1]
        token_ids = sparse_csr.indices[start_index:end_index].tolist()
        weights = sparse_csr.data[start_index:end_index].tolist()
        return dict(zip(token_ids, weights))

    def _validate_inputs(self, state: ImportGraphState) -> List[Dict[str, Any]]:
        """校验输入，返回chunks列表"""
        chunks = state.get('chunks', [])
        if not chunks or not isinstance(chunks, list):
            raise ValidationError("chunks 为空或类型无效", self.name)

        self.logger.info(f'嵌入的块数：{len(chunks)}')
        return chunks


if __name__ == '__main__':
    import json

    setup_logging()

    base_dir = Path(
        r"W:\test\PythonProject\smart_audiobook\knowledge\processor\import_processor\temp_dir"
    )
    input_path = base_dir / "chunks_with_bookname.json"
    output_path = base_dir / "chunks_vector.json"

    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)

    node = EmbeddingChunksNode()
    result_state = node.process({"chunks": chunks_data.get('chunks', chunks_data)})

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_state, f, ensure_ascii=False, indent=4)

    print(f"向量生成完成，结果已保存至:\n{output_path}")