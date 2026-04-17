import asyncio
import json
from json import JSONDecodeError
from typing import Tuple, List, Dict, Any, Union

from agents.mcp import MCPServerStreamableHttp

from knowledge.processor.query_processor.base import BaseNode
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.processor.query_processor.exceptions import StateFieldError


class WebMcpSearchNode(BaseNode):
    """
    联网搜索节点（听书平台版本）
    用于搜索书籍相关的补充信息，如：
    - 作者生平
    - 书籍评价
    - 相关新闻
    - 读者讨论
    """
    name = 'web_mcp_search_node'

    def process(self, state: QueryGraphState) -> Union[QueryGraphState, Dict[str, Any]]:
        # 1. 参数校验
        rewritten_query, book_names = self._validate_state(state)

        # 2. 构建搜索查询（结合书名）
        search_query = self._build_search_query(rewritten_query, book_names)

        # 3. 执行MCP联网搜索
        web_search_results = asyncio.run(self._execute_mcp_server(search_query))

        # 4. 判断非空
        if not web_search_results:
            self.logger.info(f"联网搜索无结果，查询: {search_query}")
            return {"web_search_docs": []}

        # 5. 格式化搜索结果
        formatted_results = self._format_search_results(web_search_results)

        # 6. 更新state
        state['web_search_docs'] = formatted_results
        self.logger.info(f"联网搜索完成，返回 {len(formatted_results)} 条结果")

        return {"web_search_docs": formatted_results}

    def _validate_state(self, state: QueryGraphState) -> Tuple[str, List[str]]:
        """校验state中的必要参数"""
        # 获取重写后的查询
        rewritten_query = state.get('rewritten_query', '')
        if not rewritten_query or not isinstance(rewritten_query, str):
            raise StateFieldError(
                node_name=self.name,
                field_name='rewritten_query',
                expected_type=str,
                message='rewritten_query不能为空'
            )

        # 获取确认的书名列表
        book_names = state.get('book_names', [])
        if not isinstance(book_names, list):
            book_names = []

        return rewritten_query, book_names

    def _build_search_query(self, rewritten_query: str, book_names: List[str]) -> str:
        """
        构建搜索查询字符串
        结合书名和用户问题，生成更精准的搜索词
        Args:
            rewritten_query: 重写后的用户问题
            book_names: 确认的书名列表
        Returns:
            搜索查询字符串
        """
        if book_names:
            book_str = ' '.join(book_names)
            # 如果有书名，优先使用书名 + 问题关键词
            return f"{book_str} {rewritten_query}"
        else:
            return rewritten_query

    async def _execute_mcp_server(self, search_query: str) -> List[Dict[str, Any]]:
        """
        执行MCP联网搜索
        Args:
            search_query: 搜索查询字符串
        Returns:
            搜索结果列表
        """
        try:
            async with MCPServerStreamableHttp(
                name='联网搜索',
                params={
                    "url": self.config.mcp_dashscope_base_url,
                    "headers": {
                        "Authorization": f"Bearer {self.config.openai_api_key}"
                    },
                    "timeout": 60
                },
                max_retry_attempts=3,
                cache_tools_list=True,
            ) as mcp_client:
                # 调用搜索工具
                web_search_result = await mcp_client.call_tool(
                    tool_name='bailian_web_search',
                    arguments={
                        "query": search_query,
                        "count": getattr(self.config, 'web_search_count', 5),
                    }
                )

                # 获取文本内容
                if not web_search_result.content:
                    self.logger.warning("MCP搜索返回内容为空")
                    return []

                text_content = web_search_result.content[0]
                if not text_content or not text_content.text:
                    self.logger.warning("MCP搜索返回文本为空")
                    return []

                # 解析JSON
                try:
                    text_content_obj = json.loads(text_content.text)
                    pages = text_content_obj.get('pages', [])
                    return pages if pages else []
                except JSONDecodeError as e:
                    self.logger.error(f'JSON解析失败: {e.msg}, 内容: {e.doc[:200]}')
                    return []

        except Exception as e:
            self.logger.error(f"MCP联网搜索执行失败: {str(e)}")
            return []

    def _format_search_results(self, raw_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        格式化搜索结果，统一输出格式
        Args:
            raw_results: 原始搜索结果
        Returns:
            格式化后的结果列表
        """
        formatted_results = []
        for page in raw_results:
            if not page:
                continue

            formatted_results.append({
                'snippet': page.get('snippet', ''),
                'title': page.get('title', ''),
                'url': page.get('url', ''),
                'source': 'web',  # 标记来源
                'content_type': 'web_search',  # 内容类型
            })

        return formatted_results


if __name__ == '__main__':
    # 测试代码（config 由 BaseNode 自动提供）
    node = WebMcpSearchNode()

    mock_state = {
        "rewritten_query": "这本书的文学价值是什么？",
        "book_names": ["活着"]
    }

    result = node.process(mock_state)

    docs = result.get("web_search_docs", [])
    print(f"\n{'='*60}")
    print(f"联网搜索结果: {len(docs)} 条")
    print(f"{'='*60}")

    for i, doc in enumerate(docs, 1):
        print(f"\n[{i}] 标题: {doc.get('title')}")
        print(f"    来源: {doc.get('url')}")
        print(f"    摘要: {doc.get('snippet', '')[:150]}...")