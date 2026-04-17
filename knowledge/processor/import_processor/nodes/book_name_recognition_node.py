import json
from typing import List, Tuple, Dict, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pymilvus import MilvusClient, DataType

from knowledge.processor.import_processor.base import BaseNode, setup_logging
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.processor.import_processor.exceptions import StateFieldError, ValidationError
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.prompts.import_prompt import BOOK_NAME_SYSTEM_PROMPT, BOOK_NAME_USER_PROMPT_TEMPLATE


class BookNameRecognitionNode(BaseNode):
    """
    书名识别节点
    1. 优先检查state中是否已有书名（来自元数据）
    2. 如果没有，利用LLM从文档内容中提取书名
    3. 生成混合向量（稠密+稀疏）并存入Milvus（按书名去重，每个书名只存一次）
    4. 回填书名到每个chunk和state
    """
    name = 'book_name_recognition_node'

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """核心方法：识别书名并存储"""
        # 1. 参数校验
        file_title, chunks, book_name_chunk_k = self._validate_state(state)

        # 3. 没有预设书名，使用LLM提取
        self.logger.info(f"开始LLM提取书名，文档标题: {file_title}")

        # 3.1 构建LLM上下文
        final_context = self._prepare_llm_context(chunks, book_name_chunk_k)

        if not final_context:
            self.logger.warning(f"无法构建LLM上下文，降级使用文件标题: {file_title}")
            book_name = file_title
        else:
            # 3.2 调用LLM提取书名
            book_name = self._recognition_book_name(final_context, file_title)

        # 4. 检查是否已存储过，未存储才入库
        if not self._book_name_exists(book_name):
            dense_vector, sparse_vector = self._embedding_book_name(book_name)
            if dense_vector and sparse_vector:
                self._insert_milvus(dense_vector, sparse_vector, file_title, book_name)
            else:
                self.logger.warning(f"书名向量化失败，跳过Milvus存储: {book_name}")
        else:
            self.logger.info(f"书名 {book_name} 已存在于Milvus，跳过存储")

        # 5. 回填到chunks和state
        self._fill_book_name(state, chunks, book_name)

        return state

    def _book_name_exists(self, book_name: str) -> bool:
        """
        检查Milvus中是否已存在该书名的记录（按book_name去重）
        Args:
            book_name: 书名
        Returns:
            True: 已存在, False: 不存在
        """
        if not book_name or book_name == 'UNKNOWN':
            return False

        try:
            client = StorageClients.get_milvus_client()
        except ConnectionError as e:
            self.logger.error(f"Milvus客户端创建失败: {str(e)}")
            return False

        collection_name = getattr(self.config, 'book_name_collection', 'book_name_collection')

        # 检查集合是否存在
        if not client.has_collection(collection_name):
            return False

        try:
            # 按book_name精确查询
            result = client.query(
                collection_name=collection_name,
                filter=f'book_name == "{book_name}"',
                output_fields=["book_name"],
                limit=1
            )
            exists = len(result) > 0
            if exists:
                self.logger.info(f"书名 {book_name} 已存在于Milvus")
            return exists
        except Exception as e:
            self.logger.warning(f"查询Milvus失败: {str(e)}")
            return False

    def _embed_and_store(self, book_name: str, file_title: str) -> None:
        """
        仅做向量化和存储（用于已有书名的情况）
        Args:
            book_name: 书名
            file_title: 文档标题
        """
        dense_vector, sparse_vector = self._embedding_book_name(book_name)
        if dense_vector and sparse_vector:
            self._insert_milvus(dense_vector, sparse_vector, file_title, book_name)
        else:
            self.logger.warning(f"书名向量化失败，跳过Milvus存储: {book_name}")

    def _fill_book_name(self, state: ImportGraphState, chunks: List[Dict], book_name: str) -> None:
        """
        回填书名到chunks和state
        Args:
            state: 节点状态
            chunks: 切片列表
            book_name: 识别的书名
        """
        # 1. 更新每个chunk的book_name
        for chunk in chunks:
            chunk['book_name'] = book_name
        # 2. 更新state
        state['book_name'] = book_name
        state['chunks'] = chunks
        self.logger.info(f"书名回填完成: {book_name}，共更新 {len(chunks)} 个chunk")

    def _insert_milvus(self, dense_vector: List, sparse_vector: Dict[str, Any],
                       file_title: str, book_name: str) -> None:
        """
        将识别的书名保存到Milvus数据库中
        数据字段:
        - pk: 主键（自增）
        - dense_vector: 稠密向量
        - sparse_vector: 稀疏向量
        - file_title: 文档标题
        - book_name: 书名
        """
        if not dense_vector or not sparse_vector:
            self.logger.warning("稠密向量或稀疏向量为空，跳过Milvus插入")
            return

        # 获取Milvus客户端
        try:
            client = StorageClients.get_milvus_client()
        except ConnectionError as e:
            self.logger.error(f"Milvus客户端创建失败: {str(e)}")
            return

        # 获取集合名（从config获取）
        book_name_collection_name = getattr(self.config, 'item_name_collection', 'book_name_collection')

        try:
            # 创建集合（幂等）
            if not client.has_collection(book_name_collection_name):
                self._create_collection_with_name(client, book_name_collection_name)

            # 构建数据行
            book_name_data_row = {
                'file_title': file_title,
                'book_name': book_name,
                'dense_vector': dense_vector,
                'sparse_vector': sparse_vector
            }

            # 插入数据
            inserted_result = client.insert(
                collection_name=book_name_collection_name,
                data=[book_name_data_row]
            )
            self.logger.info(f"书名插入Milvus成功: {book_name}，主键: {inserted_result.get('ids')}")
        except Exception as e:
            self.logger.error(f"Milvus插入数据失败: {str(e)}")

    def _create_collection_with_name(self, client: MilvusClient, collection_name: str) -> None:
        """
        创建书名集合
        Args:
            client: Milvus客户端
            collection_name: 集合名
        """
        # 1. 创建schema约束
        schema = client.create_schema()
        schema.add_field(field_name='pk', datatype=DataType.VARCHAR,
                         is_primary=True, auto_id=True, max_length=100)
        schema.add_field(field_name='file_title', datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name='dense_vector', datatype=DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field(field_name='sparse_vector', datatype=DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field(field_name='book_name', datatype=DataType.VARCHAR, max_length=65535)

        # 2. 创建索引
        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name='dense_vector',
            index_name='dense_vector_index',
            index_type='AUTOINDEX',
            metric_type='COSINE'
        )
        index_params.add_index(
            field_name='sparse_vector',
            index_name='sparse_vector_index',
            index_type='SPARSE_INVERTED_INDEX',
            metric_type='IP'
        )

        # 3. 创建集合
        client.create_collection(
            collection_name=collection_name,
            schema=schema,
            index_params=index_params
        )
        self.logger.info(f"创建Milvus集合成功: {collection_name}")

    def _embedding_book_name(self, book_name: str) -> Tuple[Optional[List], Optional[Dict[str, Any]]]:
        """
        调用BGE-M3模型获取嵌入向量
        Args:
            book_name: 书名
        Returns:
            (dense_vector, sparse_vector) 元组，失败时返回 (None, None)
        """
        if not book_name or book_name == 'UNKNOWN':
            self.logger.warning(f"书名为空或UNKNOWN，跳过向量化: {book_name}")
            return None, None

        # 获取嵌入模型客户端
        try:
            client = AIClients.get_bge_m3_client()
        except ConnectionError as e:
            self.logger.error(f"BGE-M3模型客户端获取失败: {str(e)}")
            return None, None

        try:
            # 计算稠密和稀疏向量
            vector_result = client.encode_documents([book_name])

            # 获取稠密向量
            dense_vector = vector_result['dense'][0].tolist()

            # 获取稀疏向量（CSR格式转换）
            sparse_csr = vector_result['sparse']
            start_index = sparse_csr.indptr[0]
            end_index = sparse_csr.indptr[1]
            token_ids = sparse_csr.indices[start_index:end_index].tolist()
            weights = sparse_csr.data[start_index:end_index].tolist()
            sparse_vector = dict(zip(token_ids, weights))

            self.logger.info(f"向量化成功 - 书名: {book_name}, 稠密向量维度: {len(dense_vector)}")
            return dense_vector, sparse_vector
        except Exception as e:
            self.logger.error(f"向量化失败: {str(e)}")
            return None, None

    def _validate_state(self, state: ImportGraphState) -> Tuple[str, List[Dict[str, Any]], int]:
        """
        校验state中的必要参数
        Returns:
            (file_title, chunks, book_name_chunk_k)
        """
        # 获取文档标题
        file_title = state.get('file_title', '')
        if not file_title:
            raise StateFieldError(
                node_name=self.name,
                field_name='file_title',
                expected_type=str,
                message='文档标题不能为空'
            )

        # 获取chunks
        chunks = state.get('chunks', [])
        if not chunks or not isinstance(chunks, list):
            raise StateFieldError(
                node_name=self.name,
                field_name='chunks',
                expected_type=list,
                message='chunks不能为空且必须是列表类型'
            )

        # 获取配置的chunk数量（默认3）
        book_name_chunk_k = getattr(self.config, 'book_name_chunk_k', 3)
        if book_name_chunk_k <= 0:
            raise ValidationError(message='book_name_chunk_k配置必须大于0')

        return file_title, chunks, book_name_chunk_k

    def _prepare_llm_context(self, chunks: List[Dict], k: int) -> str:
        """
        准备给LLM的上下文内容
        优先使用 content_type 为"书籍简介"的chunk
        Args:
            chunks: 切片列表
            k: 最多使用的切片数量
        Returns:
            拼接后的上下文字符串
        """
        contexts = []

        # 优先找 content_type == "书籍简介" 的chunk
        for chunk in chunks[:k]:
            content_type = chunk.get('content_type', '')
            content = chunk.get('content', '')
            if content_type == '书籍简介' and content:
                contexts.append(content)

        # 如果没有书籍简介，就用前k个chunk的content
        if not contexts:
            for chunk in chunks[:k]:
                content = chunk.get('content', '')
                if content:
                    contexts.append(content)

        if not contexts:
            self.logger.warning("未找到有效的上下文内容")
            return ""

        # 限制总长度（避免token超限，约4000字符）
        full_context = '\n\n'.join(contexts)
        if len(full_context) > 4000:
            full_context = full_context[:4000]
            self.logger.info(f"上下文过长，已截断至4000字符")

        self.logger.info(f"LLM上下文准备完成，使用了 {len(contexts)} 个切片")
        return full_context

    def _recognition_book_name(self, context: str, file_title: str) -> str:
        """
        调用LLM识别书名
        Args:
            context: 文档上下文
            file_title: 文件标题（兜底用）
        Returns:
            识别的书名
        """
        # 获取LLM客户端
        try:
            llm_client: ChatOpenAI = AIClients.get_llm_client(response_format=False)
        except ConnectionError as e:
            self.logger.error(f"LLM客户端创建失败，降级使用文件标题: {file_title}, 错误: {str(e)}")
            return file_title

        # 构建prompt（从外部导入）
        system_prompt = BOOK_NAME_SYSTEM_PROMPT
        user_prompt = BOOK_NAME_USER_PROMPT_TEMPLATE.format(
            file_title=file_title,
            context=context
        )

        try:
            llm_response = llm_client.invoke([
                SystemMessage(system_prompt),
                HumanMessage(user_prompt)
            ])
            book_name = llm_response.content.strip()

            # 校验识别结果
            if not book_name or book_name == 'UNKNOWN':
                self.logger.warning(f"LLM未能识别书名（返回: {book_name}），降级使用文件标题: {file_title}")
                return file_title

            self.logger.info(f"LLM识别成功 - 文档: {file_title} → 书名: {book_name}")
            return book_name
        except Exception as e:
            self.logger.error(f"LLM调用失败: {str(e)}，降级使用文件标题: {file_title}")
            return file_title


if __name__ == '__main__':
    import json
    from pathlib import Path

    setup_logging()

    # 读取测试JSON文件
    json_path = Path(r"W:\test\PythonProject\smart_audiobook\knowledge\processor\import_processor\temp_dir\chunks.json")

    if not json_path.exists():
        print(f"文件不存在: {json_path}")
        exit(1)

    with open(json_path, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)

    # 兼容两种格式：直接是列表 或 包含chunks字段的字典
    if isinstance(chunks_data, dict) and 'chunks' in chunks_data:
        test_chunks = chunks_data['chunks']
    elif isinstance(chunks_data, list):
        test_chunks = chunks_data
    else:
        print(f"不支持的JSON格式: {type(chunks_data)}")
        exit(1)

    # 获取文件标题（从JSON中获取，或使用默认值）
    file_title = chunks_data.get('file_title', '测试文档') if isinstance(chunks_data, dict) else '测试文档'

    test_state = {
        "file_title": file_title,
        "chunks": test_chunks,
        "book_name": ""  # 空，触发LLM识别
    }

    node = BookNameRecognitionNode()
    result = node.process(test_state)

    print(f"\n{'=' * 60}")
    print(f"识别结果: {result.get('book_name')}")
    print(f"{'=' * 60}")
    print(f"Chunks中的book_name:")
    for i, chunk in enumerate(result.get('chunks', [])):
        print(f"  [{i}] {chunk.get('book_name', 'N/A')} - {chunk.get('content_type', 'unknown')}")
    print(f"{'=' * 60}")

    # 可选：保存结果到新文件
    output_path = json_path.parent / "chunks_with_bookname.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"\n结果已保存至: {output_path}")