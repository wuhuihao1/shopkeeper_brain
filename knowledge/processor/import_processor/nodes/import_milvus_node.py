import os
import json
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


# 新项目标量字段定义（适配听书知识库）
_SCALAR_FIELDS: tuple[_SCALAR_FIELD_SPC, ...] = (
    _SCALAR_FIELD_SPC(field_name="content_type", datatype=DataType.VARCHAR, max_length=100),
    _SCALAR_FIELD_SPC(field_name="book_name", datatype=DataType.VARCHAR, max_length=500),
    _SCALAR_FIELD_SPC(field_name="author_name", datatype=DataType.VARCHAR, max_length=500),
    _SCALAR_FIELD_SPC(field_name="entry_name", datatype=DataType.VARCHAR, max_length=500),
    _SCALAR_FIELD_SPC(field_name="title", datatype=DataType.VARCHAR, max_length=500),
    _SCALAR_FIELD_SPC(field_name="file_title", datatype=DataType.VARCHAR, max_length=500),
    _SCALAR_FIELD_SPC(field_name="content", datatype=DataType.VARCHAR, max_length=65535),
    _SCALAR_FIELD_SPC(field_name="suitable_for", datatype=DataType.VARCHAR, max_length=65535),
    # 复杂类型字段存为JSON字符串
    _SCALAR_FIELD_SPC(field_name="category_tags", datatype=DataType.VARCHAR, max_length=1000),
    _SCALAR_FIELD_SPC(field_name="highlights", datatype=DataType.VARCHAR, max_length=65535),
    _SCALAR_FIELD_SPC(field_name="faq", datatype=DataType.VARCHAR, max_length=65535),
    _SCALAR_FIELD_SPC(field_name="source_file", datatype=DataType.VARCHAR, max_length=500),
    _SCALAR_FIELD_SPC(field_name="source_path", datatype=DataType.VARCHAR, max_length=1000),
)


class _MilvusSchemaBuilder:
    """负责处理和Milvus字段约束相关的逻辑"""

    @staticmethod
    def build_schema(milvus_client: MilvusClient, dim: int):
        """创建schema"""
        schema = milvus_client.create_schema(enable_dynamic_field=True)

        # 添加主键字段
        schema.add_field(field_name="chunk_id", datatype=DataType.INT64, auto_id=True, is_primary=True)

        # 添加向量字段
        schema.add_field(field_name="dense_vector", datatype=DataType.FLOAT_VECTOR, dim=dim)
        schema.add_field(field_name="sparse_vector", datatype=DataType.SPARSE_FLOAT_VECTOR)

        # 添加标量字段
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
    """负责插入Milvus数据"""

    def __init__(self, milvus_client: MilvusClient, collection_name: str):
        self._milvus_client = milvus_client
        self._collection_name = collection_name

    def insert_rows(self, data: List[Dict[str, Any]]):
        """插入数据并回填chunk_id"""
        # 预处理数据：将复杂类型转为JSON字符串
        processed_data = self._prepare_data(data)

        vector_result = self._milvus_client.insert(
            collection_name=self._collection_name,
            data=processed_data
        )
        chunk_ids = vector_result['ids']

        # 回填chunk_id到原始chunk
        for chunk_id, chunk in zip(chunk_ids, data):
            chunk['chunk_id'] = chunk_id

    def _prepare_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """将复杂类型字段（list/dict）转为JSON字符串"""
        processed = []
        for chunk in data:
            row = chunk.copy()
            # 转换列表/字典类型字段为JSON字符串
            if 'category_tags' in row and isinstance(row['category_tags'], (list, dict)):
                row['category_tags'] = json.dumps(row['category_tags'], ensure_ascii=False)
            if 'highlights' in row and isinstance(row['highlights'], (list, dict)):
                row['highlights'] = json.dumps(row['highlights'], ensure_ascii=False)
            if 'faq' in row and isinstance(row['faq'], (list, dict)):
                row['faq'] = json.dumps(row['faq'], ensure_ascii=False)

            # 确保必要字段存在
            for field in ['content_type', 'book_name', 'author_name', 'entry_name',
                          'title', 'file_title', 'content', 'suitable_for',
                          'category_tags', 'highlights', 'faq', 'source_file', 'source_path']:
                if field not in row:
                    row[field] = ""
            processed.append(row)
        return processed


class _MilvusIndexBuilder:
    """构建索引类"""

    @staticmethod
    def build_index_params(milvus_client: MilvusClient):
        index_params = milvus_client.prepare_index_params()
        # 稠密向量索引
        index_params.add_index(
            field_name='dense_vector',
            index_name='dense_vector_index',
            index_type='AUTOINDEX',
            metric_type="COSINE"
        )
        # 稀疏向量索引
        index_params.add_index(
            field_name='sparse_vector',
            index_name='sparse_vector_index',
            index_type='SPARSE_INVERTED_INDEX',
            metric_type="IP"
        )
        return index_params


class ImportMilvusNode(BaseNode):
    """门面类，用于将已向量化的数据插入到milvus中"""
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

        # 3. 获取chunks集合名（从config获取）
        chunks_collection = getattr(self.config, 'chunks_collection', 'book_chunks')
        # 4. 创建集合
        self._create_chunks_collection(chunks_collection, milvus_client, dim)

        # 5. 插入数据
        inserter = _MilvusInserter(milvus_client, chunks_collection)
        inserter.insert_rows(validated_chunks)
        self.logger.info(f'向量插入到milvus成功!')

        # 6. 备份文件
        self._back_up(validated_chunks, state)

        return state

    def _back_up(self, final_chunks: List[Dict[str, Any]], state: ImportGraphState) -> None:
        """对chunks进行备份，存放到json文件"""
        local_dir = state.get("file_dir", "")
        if not local_dir:
            return
        try:
            os.makedirs(local_dir, exist_ok=True)
            output_path = os.path.join(local_dir, "chunks_vector.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(final_chunks, f, ensure_ascii=False, indent=2)
            self.logger.info('向量数据备份成功!')
        except Exception as e:
            self.logger.warning(f"备份失败: {e}")

    def _validate_state(self, state: ImportGraphState) -> Tuple[List[Dict[str, Any]], int]:
        """校验state，返回有效的chunks和向量维度"""
        chunks = state.get('chunks', [])
        if not chunks or not isinstance(chunks, list):
            raise StateFieldError(node_name=self.name, field_name='chunks', expected_type=list)

        validated_chunks = []
        for i, chunk in enumerate(chunks):
            if not isinstance(chunk, dict):
                raise ValidationError(f'{chunk}类型无效,期望类型为字典,实际为{type(chunk).__name__}')

            dense_vector = chunk.get('dense_vector')
            sparse_vector = chunk.get('sparse_vector')

            if dense_vector and sparse_vector:
                validated_chunks.append(chunk)
            else:
                self.logger.warning(f"chunks[{i}] 缺少混合向量，已跳过")

        if not validated_chunks:
            raise ValidationError('所有 chunk 均无有效向量，无法入库', self.name)

        dim = len(validated_chunks[0]['dense_vector'])
        self.logger.info(f"有效 chunks：{len(validated_chunks)}，向量维度：{dim}")
        return validated_chunks, dim

    def _create_chunks_collection(self, chunks_collection: str, milvus_client: MilvusClient, dim: int):
        """创建集合（如果不存在）"""
        if milvus_client.has_collection(collection_name=chunks_collection):
            self.logger.info(f"{chunks_collection}已存在，跳过创建")
            return

        schema = _MilvusSchemaBuilder.build_schema(milvus_client, dim)
        index_params = _MilvusIndexBuilder.build_index_params(milvus_client)
        milvus_client.create_collection(
            collection_name=chunks_collection,
            schema=schema,
            index_params=index_params
        )
        self.logger.info(f"创建集合成功: {chunks_collection}")


def _cli_main() -> None:
    import json
    from pathlib import Path
    setup_logging()

    temp_dir = Path(
        r"W:\test\PythonProject\smart_audiobook\knowledge\processor\import_processor\temp_dir"
    )
    input_path = temp_dir / "chunks_vector.json"
    output_path = temp_dir / "chunks_vector_ids.json"

    if not input_path.exists():
        raise FileNotFoundError(f"找不到输入文件: {input_path}")

    with open(input_path, "r", encoding="utf-8") as f:
        content = json.load(f)

    state: ImportGraphState = {"chunks": content.get('chunks', content)}

    node = ImportMilvusNode()
    result_state = node.process(state)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result_state, f, ensure_ascii=False, indent=4)

    print(f"结果已保存至: {output_path}")


if __name__ == '__main__':
    _cli_main()