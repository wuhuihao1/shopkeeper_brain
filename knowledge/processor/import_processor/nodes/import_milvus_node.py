import os, json
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional
from pymilvus import MilvusClient, DataType
from knowledge.processor.import_processor.base import BaseNode, setup_logging
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.processor.import_processor.exceptions import StateFieldError, ValidationError, MilvusError
from knowledge.utils.client.storage_clients import StorageClients

@dataclass
class _SCALAR_FIELD_SPC:
    field_name: str
    datatype: DataType
    max_length: Optional[int] = None

_SCALAR_FIELDS: tuple[_SCALAR_FIELD_SPC, ...] = (
    _SCALAR_FIELD_SPC(field_name="title", datatype=DataType.VARCHAR, max_length=65535),
    _SCALAR_FIELD_SPC(field_name="item_name", datatype=DataType.VARCHAR, max_length=65535),
    _SCALAR_FIELD_SPC(field_name="parent_title", datatype=DataType.VARCHAR, max_length=65535),
    _SCALAR_FIELD_SPC(field_name="file_title", datatype=DataType.VARCHAR, max_length=65535),
    _SCALAR_FIELD_SPC(field_name="content", datatype=DataType.VARCHAR, max_length=65535),
)

class _MilvusSchemaBuilder:
    """
    负责处理和Milvus字段约束相关的逻辑
    """
    @staticmethod
    def build_schema(milvus_client: MilvusClient, dim: int):
        """
        创建schema
        Args:
            milvus_client:
            dim:

        Returns:

        """
        # 1. 创建schema
        #enable_dynamic_field启用动态字段功能
        schema = milvus_client.create_schema(enable_dynamic_field=True)
        # 2. 添加字段约束
        # 2.1 添加主键字段的约束
        schema.add_field(field_name="chunk_id", datatype=DataType.INT64, auto_id=True, is_primary=True)
        # 2.2 添加向量字段的约束
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)
        # 2.3 遍历_SCALAR_FIELD_SPC添加标量字段的约束[标量字段个数：5个]
        #有的字段有max_length有的没有,所以要做一个判断
        for spec in _SCALAR_FIELDS:
            kwargs: Dict = {
                "field_name": spec.field_name,
                "datatype": spec.datatype,
            }
            if spec.max_length:
                kwargs["max_length"] = spec.max_length
            schema.add_field(**kwargs)
        return schema

class _MilvusInserter:
    """
    负责插入Milvus数据
    """
    def __init__(self, milvus_client: MilvusClient, collection_name: str):
        self._milvus_client = milvus_client
        self._collection_name = collection_name
    def insert_rows(self, data: List[Dict[str, Any]]):

        # 1. 插入
        vector_result = self._milvus_client.insert(collection_name=self._collection_name, data=data)
        # 2. 得到每一个chunk的id
        chunk_ids = vector_result['ids']
        # 3. 回填到chunk中
        #data是一个由chunk组成的List,chunk_ids也是个由chunk_id组成的list
        #所以用zip函数缝合然后遍历
        for chunk_id, chunk in zip(chunk_ids, data):
            chunk['chunk_id'] = chunk_id

class _MilvusIndexBuilder:
    """构建索引类"""

    @staticmethod
    def build_index_params(milvus_client: MilvusClient):
        #构建自定义index对象
        index_params = milvus_client.prepare_index_params()
        # 稠密向量：AUTOINDEX
        index_params.add_index(
            field_name='dense_vector',
            index_name='dense_vector_index',
            index_type='AUTOINDEX',
            metric_type="COSINE"
        )
        # 稀疏向量：倒排索引SPARSE_INVERTED_INDEX
        index_params.add_index(
            field_name='sparse_vector',
            index_name='sparse_vector_index',
            index_type='SPARSE_INVERTED_INDEX',
            metric_type="IP"
        )
        return index_params

class ImportMilvusNode(BaseNode):
    """
    门面类,用于将已向量化的数据插入到milvus中
    """
    name = "import_milvus_node"

    def process(self, state: ImportGraphState) -> ImportGraphState:
        self.logger.info(f'开始插入向量到数据库milvus')
        # 1. 校验state
        validated_chunks, dim = self._validate_state(state)
        # 2. 获取Milvus客户端
        try:
            milvus_client = StorageClients.get_milvus_client()
        except ConnectionError as e:
            self.logger.error(f'连接到milvus失败,原因:{e}')
            raise MilvusError(f"MilVus客户端创建失败,异常原因{str(e)}", node_name=self.name)
        # 3. 获取chunks集合
        chunks_collection = self.config.chunks_collection
        # 4. 创建集合
        self._create_chunks_collection(chunks_collection, milvus_client, dim)
        # 5. 插入数据(静态字段都要提供)
        inserter = _MilvusInserter(milvus_client, chunks_collection)
        inserter.insert_rows(validated_chunks)
        self.logger.info(f'向量插入到milvus成功!')
        #备份文件
        self._back_up(validated_chunks, state)
        # 6. 返回state
        return state

    def _back_up(self, final_chunks: List[Dict[str, Any]], state: ImportGraphState) -> None:
        """
        对chunks进行备份,存放到json文件
        Args:
            final_chunks:最后组装好的chunk
            state:节点状态

        Returns:

        """
        #获取存放目录
        local_dir = state.get("file_dir", "")
        if not local_dir:
            return
        try:
            #exist_ok属性,开启后允许路径已经存在文件夹
            os.makedirs(local_dir, exist_ok=True)
            #拼装路径
            output_path = os.path.join(local_dir, "chunks_vector.json")
            # 写入文件
            with open(output_path, "w", encoding="utf-8") as f:
                #允许输出中文ensure_ascii
                json.dump(final_chunks, f, ensure_ascii=False, indent=2)
            self.logger.info('向量数据备份成功!')
        except Exception as e:
            self.logger.warning(f"备份失败: {e}")

    def _validate_state(self, state: ImportGraphState)-> Tuple[List[Dict[str, Any]], int]:
        chunks = state['chunks']
        if not chunks or not isinstance(chunks, list):
            raise StateFieldError(node_name=self.name, field_name='chunks', expected_type=list)

        validated_chunks = []
        for i, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                raise ValidationError(f'{chunk}类型无效,期望类型为字典,实际为{type(chunk).__name__}')

            dense_vector = chunk['dense_vector']
            sparse_vector = chunk['sparse_vector']
            if dense_vector and sparse_vector:
                validated_chunks.append(chunk)
            else:
                self.logger.warning(f"chunks[{i}] 缺少混合向量，已跳过")
        if not validated_chunks:
            raise ValidationError('所有 chunk 均无有效向量，无法入库', self.name)
        dim = len(chunks[0]['dense_vector'])
        self.logger.info(f"有效 chunks：{len(validated_chunks)}，向量维度：{dim}")
        return validated_chunks, dim

    def _create_chunks_collection(self, chunks_collection: str, milvus_client: MilvusClient, dim: int):
        # 1. 判断集合是否存在
        if milvus_client.has_collection(collection_name=chunks_collection):
            self.logger.info(f"{chunks_collection}已存在 跳过创建")
            return
        # 2. 创建schema
        schema = _MilvusSchemaBuilder.build_schema(milvus_client, dim)
        # 3. 创建索引
        index_params = _MilvusIndexBuilder.build_index_params(milvus_client)
        # 4. 创建集合
        milvus_client.create_collection(collection_name=chunks_collection, schema=schema, index_params=index_params)

def _cli_main() -> None:
    import json
    from pathlib import Path
    setup_logging()

    temp_dir = Path(
        r"W:\test\PythonProject\shopkeeper_brain\knowledge\processor\import_processor\temp_dir"
        )
    input_path = temp_dir / "chunks_vector.json"
    output_path = temp_dir / "chunks_vector_ids.json"

    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        content = json.load(f)

    state: ImportGraphState = {"chunks": content.get('chunks')}

    node = ImportMilvusNode()
    result_state = node.process(state)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_state, f, ensure_ascii=False, indent=4)

    print(f"结果已保存至: {output_path}")


if __name__ == '__main__':
    _cli_main()



