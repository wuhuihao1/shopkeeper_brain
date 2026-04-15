import asyncio
import json
from json import JSONDecodeError
from typing import Tuple, List, Dict, Any, Union
from agents.mcp import MCPServerStreamableHttp

from knowledge.processor.query_processor.base import BaseNode, T
from knowledge.processor.query_processor.state import QueryGraphState
from knowledge.processor.query_processor.exceptions import StateFieldError

class WebMcpSearchNode(BaseNode):
    name='web_mcp_search_node'

    def process(self, state: QueryGraphState) -> Union[QueryGraphState,Dict[str, Any]]:
        # 1. 参数校验
        rewritten_query, item_names = self._validate_state(state)
        # 2. 定义并且执行mcp的调用
        web_search_results = asyncio.run(self._execute_mcp_server(rewritten_query))
        # 3. 判断非空
        if not web_search_results:
            return {"web_search_docs": []}
        # 4. 更新state
        state['web_search_docs'] = web_search_results
        return {"web_search_docs": web_search_results}

    async def _execute_mcp_server(self, rewritten_query: str) -> List[Dict[str, Any]]:
        """
        执行MCP服务
        注意：一个MCP服务下可能有多个工具（工具:就是函数)
        Args:
            rewritten_query:
        Returns:
        """
        async with MCPServerStreamableHttp(
            name='联网搜索',
            params={
                "url":self.config.mcp_dashscope_base_url,
                "headers": {
                    "Authorization": f"Bearer {self.config.openai_api_key}"
                },
                "timeout": 60
            },
            max_retry_attempts=3,
            cache_tools_list=True,
        ) as mcp_client:
            #调用
            web_search_result = await mcp_client.call_tool(tool_name='bailian_web_search', arguments={
                "query": rewritten_query,
                "count": 3,
            })
            #一个text_content对象
            text_content = web_search_result.content[0]
            #判断非空
            if not text_content:
                return []
            text_content_text = text_content.text
            #text_content_text是一个json字符串
            if not text_content_text:
                return []
            # json反序列化
            try:
                text_content_obj = json.loads(text_content_text)
                pages = text_content_obj.get('pages', [])
                # 判断非空
                if not pages:
                    return []
                # 遍历
                web_search_results = []
                for page in pages:
                    web_search_results.append({
                        'snippet': page.get('snippet', ''),
                        'title': page.get('title', ''),
                        'url': page.get('url', ''),
                    })
                return web_search_results
            except JSONDecodeError as e:
                self.logger.error(f'web_search检索失败 失败信息：{e.msg} 失败的内容:{e.doc} 失败的位置：{e.pos}')
                return []


    def _validate_state(self, state: QueryGraphState) -> Tuple[str, List[str]]:
        # 1. 用户的问题（LLM重写后的）
        rewritten_query = state.get('rewritten_query')

        # 2. 获取商品名列表
        item_names = state.get('item_names')

        # 3. 校验
        if not rewritten_query or not isinstance(rewritten_query, str):
            raise StateFieldError(node_name=self.name, field_name='rewritten_query', expected_type=str)

        if not item_names or not isinstance(item_names, list):
            raise StateFieldError(node_name=self.name, field_name='item_names', expected_type=list)

        return rewritten_query, item_names

if __name__ == '__main__':
    web_search_node = WebMcpSearchNode()

    mock_state = {
        "rewritten_query": "今天天气如何呢",
        "item_names": ["RS-12 数字万用表"],
    }

    web_search_node.process(mock_state)
