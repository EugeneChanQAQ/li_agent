import httpx
import json

from typing import Any
from urllib.parse import urlsplit

from harness.tool.models import ToolDefinition

class ReadWebTool:
    """
    读取网页内容的工具。
    """

    def __init__(
            self,
            client: httpx.AsyncClient,
            api_key: str,
            max_chars: int = 12000,
    ) -> None:
        api_key = api_key.strip()
        if not api_key:
            raise ValueError("api_key 不能为空。")

        if max_chars <= 0:
            raise ValueError("max_chars 必须为正整数。")

        self._client = client
        self._api_key = api_key
        self._max_chars = max_chars

        self.definition = ToolDefinition(
            name="read_web",
            description=(
                "读取指定网页的内容，返回提取出的网页正文。"
                "适用于阅读搜索结果中的链接，或用户提供的网页。"
                "长正文可能被截断，请检查返回的 truncated 字段。"
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "minLength": 1,
                        "description": (
                        "要读取完整的网页 URL。"
                        "必须使用 http 或 https 协议。"
                        ),
                    },
                },
                "required": ["url"],
                "additionalProperties": False,
            },
        )

    async def execute(self, arguments: dict[str, Any]) -> str:
        url = arguments["url"].strip()
        # 涉及到 url 解析，使用 python 库 urlsplit，把 url 分成 scheme 协议、netloc 网络位置、path 路径、query 查询参数、fragment页面片段标识
        parsed_url = urlsplit(url)

        if (
            parsed_url.scheme not in ("https", "http")
            or not parsed_url.hostname
        ):
            raise ValueError("url 必须是包含主机名的 HTTP 或 HTTPS 地址")

        # 3. 请求 Tavily 提取网页正文
        response = await self._client.post(
            "https://api.tavily.com/extract",
            headers={
                "Authorization": f"Bearer {self._api_key}"
            },
            json={
                "urls": [url],
                "extract_depth": "basic",
                "format": "markdown",
                "include_images": False,
                "timeout": 10.0,
            },
            timeout=httpx.Timeout(20.0, connect=5.0),
        )

        response.raise_for_status()
        data = response.json()

        # 4. HTTP 成功返回后，继续判断网页提取是否成功
        if not isinstance(data, dict):
            raise ValueError("网页提取服务返回的数据必须是对象。")

        results = data.get("results")
        failed_results = data.get("failed_results")

        if not isinstance(results, list):
            raise ValueError("results 必须是列表")

        if not isinstance(failed_results, list):
            raise ValueError("failed_results 必须是列表。")

        # 本工具每次只请求一个 URL，出现失败记录就视为失败
        if failed_results:
            failure = failed_results[0]
            if not isinstance(failure, dict):
                raise ValueError("网页提取失败记录格式异常")

            reason = failure.get("error", "未知原因")
            raise RuntimeError(f"网页提取失败：{reason}")

        if len(results) != 1:
            raise ValueError("网页提取服务未返回预期的单个结果。")

        result = results[0]
        if not isinstance(result, dict):
            raise ValueError("网页提取结果必须是对象。")

        content = result.get("raw_content")
        result_url = result.get("url")

        if not isinstance(content, str) or not content.strip():
            raise RuntimeError("网页提取成功，但没有可用正文。")

        if not isinstance(result_url, str) or not result_url.strip():
            raise ValueError("网页提取结果缺少有效 URL。")

        # 5. 限制进入模型上下文的正文长度
        returned_content = content[:self._max_chars]

        return json.dumps(
            {
                "requested_url": url,
                "url": result_url,
                "content": returned_content,
                "extracted_chars": len(content),
                "returned_chars": len(returned_content),
                "truncated": len(content) > self._max_chars,
            },
            ensure_ascii=False,
        )

