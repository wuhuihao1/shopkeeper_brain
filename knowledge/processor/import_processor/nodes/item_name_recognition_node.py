from typing import List, Tuple, Dict, Any, Optional
from pathlib import Path
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pymilvus.model.hybrid import BGEM3EmbeddingFunction
from pymilvus import MilvusClient, DataType
from knowledge.processor.import_processor.base import BaseNode, setup_logging, T
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.processor.import_processor.exceptions import StateFieldError, ValidationError
from knowledge.utils.client.ai_clients import AIClients
from knowledge.utils.client.storage_clients import StorageClients
from knowledge.prompts.import_prompt import ITEM_NAME_SYSTEM_PROMPT, ITEM_NAME_USER_PROMPT_TEMPLATE

class ItemNameRecognitionNode(BaseNode):
    name='item_name_recognition_node'
    def process(self, state: ImportGraphState) -> ImportGraphState:
        """
        核心方法
        识别商品名称节点
        1. 负责利用LLM提取商品的具体型号（名）
        2. 嵌入商品具体型号（名）
        3. 存储到Milvus中（MySQL：模糊查询的时候不会考虑语义）
        Args:
            state: 通用节点状态

        Returns:
            更新后节点状态
        """
        # 1. 参数校验
        file_title, chunks, item_name_chunk_k = self._validate_state(state)
        # 2. 构建上下文
        final_context = self._prepare_llm_context(chunks, item_name_chunk_k)
        # 3. 调用LLM模型 提取商品名
        item_name = self._recognition_item_name(final_context, file_title)
        # 4. 向量化(嵌入模型：1.OpenAIEmbedding(OpenAI) 2.文本嵌入模型（text-embedding-v(x))（灵积服务平台：dashscope） 3.bge(bge-m3))：混合向量[稠密：相似性匹配、稀疏：精确匹配]
        dense_vector, sparse_vector = self._embedding_item_name(item_name)
        # 5.入库
        self._insert_milvus(dense_vector, sparse_vector, file_title, item_name)
        # 6.回填(更新LLM提取到的item_name)
        self._fill_item_name(state, chunks, item_name)
        return state


    def _fill_item_name(self, state: ImportGraphState, chunks: List[Dict], item_name: str):
        """
        回填item_name
        给chunk,提供下游模型使用
        给state,提供下游节点使用
        Args:
            state: 节点状态
            chunks: 切片
            item_name: llm生成的商品名称
        """
        # 1. 更新chunk的item_name
        for chunk in chunks:
            chunk['item_name'] = item_name
        state['item_name'] = item_name


    def _insert_milvus(self, dense_vector: List, sparse_vector: Dict[str, Any], file_title: str, item_name: str):
        """
        将LLM识别到的商品名保存到Milvus数据库中
        数据字段:
        pk:id
        dense_vector: 稠密向量
        sparse_vector: 稀疏向量
        file_title: 文档标题
        item_name: llm读的商品名称
        """
        # 1. 判断稠密向量和稀疏向量是否都存在
        if not dense_vector or not sparse_vector:
            return
        # 2. 获取Milvus客户端
        try:
            client = StorageClients.get_milvus_client()
        except ConnectionError as e:
            self.logger.error(f"Milvus客户端创建失败,原因：{str(e)}")
            return
        # 3. Milvus三大核心概念（集合：Collection[1.集合名 2.字段约束：schema 3.索引]）
        # 3.1 集合名：表名类似于归纳数据的容器，逻辑概念
        # 3.2 约束：类似于MySQL字段的长度、字段的类型..
        # 3.3 索引：类似于MySQL中的索引【索引类型比较多 B+树 Hash】。Milvus索引类型也有很多（专门针对于稠密向量的索引类型 针对于稀疏向量的索引类型 标量字段类型、主键类型）
        # 索引：本质就是算法（图、树、hash..）目的：提高检索【查询】效率。milvus中不管稠密向量索引还是稀疏向量的索引都是为了能够快速找到和问题相似的向量。
        # 使用Milvus的流程：①：创建集合（约束、索引）②：插入数据  ③：查询/检索
        # 4. 获取集合名字
        item_name_collection_name = self.config.item_name_collection
        try:
            # 5. 创建Milvus集合(幂等性校验)
            if not client.has_collection(item_name_collection_name):
                self._create_collection_with_name(client, item_name_collection_name)
            # 6. 构建数据行
            item_name_data_row = {
                'file_title': file_title,
                'item_name': item_name,
                'dense_vector': dense_vector,
                'sparse_vector': sparse_vector
            }
            # 7. 插入数据
            inserted_result = client.insert(collection_name=item_name_collection_name, data=[item_name_data_row])
            self.logger.info(f"插入的结果:{inserted_result},主键值:{inserted_result.get('ids')}")
        except Exception as e:
            self.logger.error(f'Milvus插入数据失败:原因{e}')



    def _create_collection_with_name(self, client: MilvusClient, item_name_collection_name: str):
        """
        创建商品名集合
        Args:
            client: 库
            item_name_collection_name: 集合名
        """
        # 1. 创建schema约束
        schema = client.create_schema()
        # 1.1 创建主键字段约束
        schema.add_field(field_name='pk', datatype=DataType.VARCHAR, is_primary=True, auto_id=True, max_length=10)
        # 1.2 创建标量字段的约束
        schema.add_field(field_name='file_title', datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name='dense_vector', datatype=DataType.FLOAT_VECTOR, dim=1024)
        schema.add_field(field_name='sparse_vector', datatype=DataType.SPARSE_FLOAT_VECTOR)
        schema.add_field(field_name='item_name', datatype=DataType.VARCHAR, max_length=65535)
        # 2. 创建索引(标量字段建立索引 向量字段建立索引)
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
        client.create_collection(collection_name=item_name_collection_name, schema = schema, index_params = index_params)

        self.logger.info(f"创建{item_name_collection_name}集合成功")


    def _embedding_item_name(self, item_name: str) -> Tuple[Optional[List], Optional[Dict[str, Any]]]:
        """
        调用bge-m3模型获取嵌入向量
        Args:
            item_name: llm吐出来的商品名

        Returns:
            稠密向量,稀疏向量

        """
        # 1. 获取到嵌入模型
        try:
            client = AIClients.get_bge_m3_client()
        except ConnectionError as e:
            self.logger.error(f'嵌入模型获取失败:{str(e)}')
            return None, None
        # 2. 计算稠密和稀疏向量
        try:
            # 3. 解析稠密向量和稀疏向量
            vector_result = client.encode_documents([item_name])
            # 3.1 获取稠密向量
            dense_vector = vector_result['dense'][0].tolist()
            # 3.2 获取稀疏向量矩阵csr(被压缩后的)三个核心属性
            #indptr 行索引
            #indices: token_id
            #data: 权重
            sparse_csr = vector_result['sparse']
            # 3.1 获取行索引
            start_index = sparse_csr.indptr[0]
            end_index = sparse_csr.indptr[1]
            # 3.2 获取token_id
            token_id = sparse_csr.indices[start_index:end_index].tolist()
            # 3.3 获取weight
            weight = sparse_csr.data[start_index:end_index].tolist()
            # 3.4 构建{"token_id":weight}字典结构
            sparse_vector = dict(zip(token_id, weight))
            self.logger.info(f'计算出来的稠密向量的维度: {len(dense_vector)}')
            return dense_vector, sparse_vector
        except Exception as e:
            self.logger.error(f'嵌入模型获取向量失败:{str(e)}')
            return None, None



    def _validate_state(self, state: ImportGraphState) -> Tuple[str, List[Dict[str, Any]], int]:
        # 1. 获取文档标题(商品具体型号[名]兜底)
        file_title = state['file_title']
        # 2. 判断文档标题
        if not file_title:
            raise StateFieldError(node_name=self.name, field_name='file_title', expected_type=str)
        # 3. 获取chunks(供LLM作为上下文信息)
        chunks = state['chunks']
        # 4. 判断chunks
        if not chunks or not isinstance(chunks, list):
            raise StateFieldError(node_name=self.name, field_name='chunks', expected_type=list)
        # 5. 获取item_name_chunk_k(3)
        item_name_chunk_k = self.config.item_name_chunk_k
        if not item_name_chunk_k or item_name_chunk_k <= 0:
            raise ValidationError(message='商品名识别的辅助切片数不合法')
        # 6. 返回
        return file_title, chunks, item_name_chunk_k

    def _prepare_llm_context(self, chunks: List[Dict], item_name_chunk_k: int) -> str:
        """
        准备给LLM大模型识别商品名的内容
        Args:
            chunks: 切片
            item_name_chunk_k: 准备使用的块数
        Returns:
            返回字符串
        """
        final_context = []
        #遍历chunks切片
        for index, chunk in enumerate(chunks[:item_name_chunk_k]):
            # 1.1 不是字典类型 直接过滤掉该块
            if not isinstance(chunk, dict):
                continue
            # 1.2 获取chunk的content
            content = chunk['content']

            slice_chunk = f'【切片】_{index}_{content}'

            final_context.append(slice_chunk)

        return '\n'.join(final_context)

    def _recognition_item_name(self, item_name_context: str, file_title: str) -> str:
        """
        调用llm识别商品名称
        Args:
            item_name_context: 名称
            file_title: 文件标题

        Returns:
            返回商品名
        """
        # 1.调用LLM客户端
        try:
            llm_client: ChatOpenAI = AIClients.get_llm_client(response_format=False)
        except ConnectionError as e:
            self.logger.error(f"OpenAI 的LLM客户端创建失败,降级使用文件标题{file_title}作为商品名 {str(e)}")
            return file_title
        # 2. 调用LLM模型
        # 2.1 获取商品名识别的llm系统提示词
        system_prompt = ITEM_NAME_SYSTEM_PROMPT
        # 2.2 获取商品名识别的llm用户提示词
        user_prompt = ITEM_NAME_USER_PROMPT_TEMPLATE.format(file_title=file_title, context=item_name_context)
        # 3. 调用 返回AIMessage对象
        try:
            llm_res = llm_client.invoke(
                [SystemMessage(system_prompt), HumanMessage(user_prompt)]
            )
            # 4. 获取AI回复的具体内容
            llm_result = llm_res.content.strip()
            if not llm_result or llm_result == 'UNKNOWN':
                self.logger.error(f"LLM提取商品名失败，降级使用文件标题{file_title}作为商品名兜底")
                return file_title
            self.logger.info(f"LLM为文档：{file_title} 提取的商品名：{llm_result}")
            return llm_result
        except Exception as e:
            self.logger.error(f"LLM提取商品名失败，降级使用文件标题{file_title}作为商品名: {str(e)}")
            return file_title

if __name__ == '__main__':
    import json

    setup_logging()

    # 1. 读取chunk.json
    temp_dir = Path(
        r"W:\test\PythonProject\shopkeeper_brain\knowledge\processor\import_processor\temp_dir")

    chunk_json_path = temp_dir / "chunks.json"
    output_path = temp_dir / "chunks_item_name.json"
    with open(chunk_json_path, "r", encoding="utf-8") as f:
        chunk_content = json.load(f)

    # 2. 构建state
    state = {
        "file_title": "万用表的使用",
        "chunks": chunk_content
    }

    # 3. 实例化节点
    node = ItemNameRecognitionNode()

    # 4. 调用process
    result = node.process(state)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=4)
    print(f"item_name:{result.get('item_name')}生成完成，结果已保存至:\n{output_path}")



