import json
from typing import Any

import httpx

from harness.tool.models import ToolDefinition

class WebSearchTool:
    def __init__(
            self,
            client: httpx.AsyncClient,
            api_key: str,
    ) -> None:
        api_key = api_key.strip()
        if not api_key:
            raise ValueError("api_key 不能为空。")

        self._client = client
        self._api_key = api_key

        # 1. 提供给模型的工具定义
        self.definition = ToolDefinition(
            name="web_search",
            description=(
                "根据搜索词查找互联网信息，"
                "返回相关网页的标题、URL 和内容摘要。"
                "需要查找资料或最新信息时使用。"
                "结果是搜索片段，不代表已经读取网页全文。"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "minLength": 1,
                        "pattern": r"^\S",
                        "description": "用于搜索的关键词或问题。",
                    },
                    "max_results": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 10,
                        "default": 5,
                        "description": "最多返回多少条结果，默认 5 条。",
                    },
                },
                "required": ["query"],
                "additionalProperties": False,
            }
        )

    async def execute(self, arguments: dict[str, Any]) -> str:
        # 参数类型和范围由 ToolExecutor 在调用前验证，这里直接使用。
        query = arguments["query"].strip()
        max_results = arguments.get("max_results", 5)

        if not query:
            raise ValueError("query 不能为空。")

        # 2.将工具参数转换为 Tavily 的 HTTP 请求
        response = await self._client.post(
            "https://api.tavily.com/search",
            headers={
                "Authorization": f"Bearer {self._api_key}",
            },
            json={
                "query": query,
                "max_results": max_results,
                "topic": "general",
                "search_depth": "basic",
                "auto_parameters": False,
                "include_answer": False,
                "include_raw_content": False,
                "include_images": False,
            },
            timeout=httpx.Timeout(20.0, connect=5.0),
        )

        # HTTP 请求失败时抛出异常，不把失败当成“没有结果”
        response.raise_for_status()

        # 3. 整理为需要的结果格式
        data = response.json()

        raw_results = data["results"]

        if not isinstance(raw_results, list):
            raise ValueError("搜索服务返回的 results 必须是列表。")

        results = [
            {
                "title": item["title"],
                "url": item["url"],
                "snippet": item["content"],
            }
            for item in raw_results[:max_results]
        ]

        return json.dumps(
            {
                "query": query,
                "results": results,
            },
            ensure_ascii=False,
        )

        