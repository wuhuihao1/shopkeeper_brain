import re, os, json
from typing import Tuple, List, Dict, Any
from pathlib import Path
from knowledge.processor.import_processor.base import BaseNode, setup_logging
from knowledge.processor.import_processor.state import ImportGraphState
from knowledge.processor.import_processor.exceptions import StateFieldError
from knowledge.dictionary.import_file import H2_TO_CONTENT_TYPE


class StandardDocParser:
    """
    新项目标准格式MD文档解析器
    格式要求：
    # 文档标题

    ## 元数据
    - 字段名：值

    ## 内容正文
    ...
    """

    def parse(self, md_content: str, source_file: str, source_path: str = "") -> List[Dict[str, Any]]:
        """解析标准文档，返回多个知识条目"""
        result = []

        # 提取元数据块
        metadata = self._extract_metadata(md_content)

        # 按二级标题切分内容块
        sections = self._split_by_h2(md_content)

        # 为每个内容块生成知识条目
        for section in sections:
            h2_title = section["title"]
            content_type = H2_TO_CONTENT_TYPE.get(h2_title)

            # 跳过元数据块
            if content_type is None:
                continue

            # 解析FAQ特殊格式
            if content_type == "常见问答":
                faq_list = self._parse_faq(section["body"])
                entry = {
                    "content": section["body"],
                    "content_type": content_type,
                    "book_name": metadata.get("书名", ""),
                    "author_name": metadata.get("作者名", ""),
                    "entry_name": metadata.get("条目名称", ""),
                    "title": section["title"],
                    "category_tags": self._parse_tags(metadata.get("类别/标签", "")),
                    "source_file": source_file,
                    "source_path": source_path,
                    "faq": faq_list,
                    "suitable_for": "",
                    "highlights": []
                }
            else:
                entry = {
                    "content": section["body"],
                    "content_type": content_type,
                    "book_name": metadata.get("书名", ""),
                    "author_name": metadata.get("作者名", ""),
                    "entry_name": metadata.get("条目名称", ""),
                    "title": section["title"],
                    "category_tags": self._parse_tags(metadata.get("类别/标签", "")),
                    "source_file": source_file,
                    "source_path": source_path,
                    "suitable_for": "",
                    "highlights": [],
                    "faq": []
                }

            # 提取扩展字段
            if content_type == "推荐运营资料":
                entry["suitable_for"] = self._extract_suitable_for(section["body"])
                entry["highlights"] = self._extract_highlights(section["body"])

            result.append(entry)

        return result

    def _extract_metadata(self, md_content: str) -> Dict[str, str]:
        """从 ## 元数据 块中提取键值对，格式：- 字段名：值"""
        metadata = {}
        pattern = r"##\s*元数据\s*\n(.*?)(?=\n##\s|\Z)"
        match = re.search(pattern, md_content, re.DOTALL)
        if match:
            block = match.group(1)
            lines = block.strip().split('\n')
            for line in lines:
                line = line.strip()
                if line.startswith('-'):
                    line = re.sub(r'^-\s*', '', line)
                    parts = re.split(r'[：:]', line, 1)
                    if len(parts) == 2:
                        key = parts[0].strip()
                        value = parts[1].strip()
                        metadata[key] = value
        return metadata

    def _split_by_h2(self, md_content: str) -> List[Dict[str, str]]:
        """按 ## 二级标题切分，保留标题作为类型标识"""
        sections = []

        # 移除一级标题行
        lines = md_content.split('\n')
        content_lines = []
        found_first_h1 = False
        for line in lines:
            if not found_first_h1 and line.strip().startswith('# '):
                found_first_h1 = True
                continue
            content_lines.append(line)
        content = '\n'.join(content_lines)

        # 按二级标题切分
        pattern = r'(##\s+[^\n]+)\n(.*?)(?=\n##\s+|\Z)'
        matches = re.findall(pattern, content, re.DOTALL)

        for match in matches:
            title_line = match[0].strip()
            title = re.sub(r'^##\s+', '', title_line).strip()
            body = match[1].strip()

            if body:
                sections.append({
                    "title": title,
                    "body": body
                })

        return sections

    def _parse_tags(self, tag_str: str) -> List[str]:
        """解析标签字符串为列表"""
        if not tag_str:
            return []
        tags = re.split(r'[,，]', tag_str)
        return [t.strip() for t in tags if t.strip()]

    def _parse_faq(self, body: str) -> List[Dict[str, str]]:
        """解析FAQ格式：问题：xxx\n回答：xxx"""
        faq_list = []
        pattern = r'问题[：:]\s*(.+?)\n回答[：:]\s*(.+?)(?=\n问题[：:]|\Z)'
        matches = re.findall(pattern, body, re.DOTALL)
        for match in matches:
            faq_list.append({
                "question": match[0].strip(),
                "answer": match[1].strip()
            })
        return faq_list

    def _extract_suitable_for(self, body: str) -> str:
        """从推荐运营资料中提取适合人群"""
        pattern = r'适合人群[：:]\s*(.+?)(?=\n\n|\n#|\Z)'
        match = re.search(pattern, body, re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    def _extract_highlights(self, body: str) -> List[str]:
        """从内容中提取核心看点"""
        highlights = []
        pattern = r'[-•*]\s*(.+?)(?=\n[-•*]|\n\n|\Z)'
        matches = re.findall(pattern, body, re.DOTALL)
        if matches:
            return [m.strip() for m in matches if len(m.strip()) > 5]
        return highlights


class DocumentSplitNode(BaseNode):
    """
    文档切分节点 - 标准格式解析
    专门用于解析包含 ## 元数据 的标准格式MD文档
    """
    name = 'document_split_node'

    def process(self, state: ImportGraphState) -> ImportGraphState:
        """文档切分的核心逻辑入口"""
        # 1. 获取并校验md_content
        md_content = self._get_md_content(state)

        file_title = state.get('file_title', '')
        source_file = state.get('source_file', file_title)
        source_path = state.get('source_path', state.get('md_path', ''))

        # 2. 使用标准解析器
        self.logger.info("使用标准格式解析器")
        parser = StandardDocParser()
        entries = parser.parse(md_content, source_file, source_path)

        if not entries:
            self.logger.warning("解析结果为空")
            state['chunks'] = []
            return state

        # 3. 转换为chunks格式
        final_chunks = self._entries_to_chunks(entries)

        # 4. 备份
        self._back_up(final_chunks, state)

        # 5. 更新状态
        state['chunks'] = final_chunks
        self.logger.info(f"标准格式解析完成，生成 {len(final_chunks)} 个知识条目")

        return state

    def _get_md_content(self, state: ImportGraphState) -> str:
        """获取md_content，如果为空则从md_path读取"""
        md_content = state.get('md_content', '')

        if not md_content:
            md_path = state.get('import_file_path', '')
            if md_path and Path(md_path).exists():
                self.logger.info(f"md_content为空，从文件读取: {md_path}")
                with open(md_path, 'r', encoding='utf-8') as f:
                    md_content = f.read()
            else:
                raise StateFieldError(
                    node_name=self.name,
                    field_name='md_content',
                    expected_type=str,
                    message='md_content为空且md_path不存在或无效'
                )

        # 统一换行符
        md_content = md_content.replace("\r\n", "\n").replace("\r", "\n")
        return md_content

    def _entries_to_chunks(self, entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """将解析后的知识条目转换为chunk格式"""
        chunks = []
        for entry in entries:
            chunk = {
                "content": entry["content"],
                "content_type": entry["content_type"],
                "book_name": entry["book_name"],
                "author_name": entry["author_name"],
                "entry_name": entry.get("entry_name", ""),
                "title": entry["title"],
                "category_tags": entry["category_tags"],
                "source_file": entry["source_file"],
                "source_path": entry.get("source_path", ""),
                "suitable_for": entry.get("suitable_for", ""),
                "highlights": entry.get("highlights", []),
                "faq": entry.get("faq", [])
            }
            chunks.append(chunk)
        return chunks

    def _back_up(self, final_chunks: List[Dict[str, Any]], state: ImportGraphState) -> None:
        """对chunks进行备份，存放到json文件"""
        local_dir = state.get("file_dir", "")
        if not local_dir:
            return
        try:
            os.makedirs(local_dir, exist_ok=True)
            output_path = os.path.join(local_dir, "chunks.json")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(final_chunks, f, ensure_ascii=False, indent=2)
            self.logger.info(f"chunks备份成功: {output_path}")
        except Exception as e:
            self.logger.warning(f"备份失败: {e}")


if __name__ == '__main__':
    setup_logging()
    document_split_node = DocumentSplitNode()

    # 测试标准格式文档
    md_content = """# 活着简介

## 元数据

- 书名：活着
- 作者名：余华
- 条目名称：全书
- 类别/标签：当代文学，现实主义，家庭

## 内容正文

《活着》通过主人公福贵一生的遭遇，书写普通人如何在一次次失去中继续活下去。

## 作者介绍

余华是中国当代重要作家之一。

## 常见问答

问题：这本书是不是特别压抑？
回答：情感上确实比较沉重。

问题：适合第一次听文学作品的人吗？
回答：适合。
"""

    init_state = {
        "md_content": md_content,
        "file_title": "活着简介",
        "file_dir": "./temp_dir",
        "source_file": "活着_完整资料.md",
        "source_path": "/data/books/活着_完整资料.md"
    }

    document_split_node.config = None
    result = document_split_node.process(init_state)

    result_str = json.dumps(result.get('chunks', []), indent=4, ensure_ascii=False)
    print("=" * 60)
    print("解析结果:")
    print("=" * 60)
    print(result_str)