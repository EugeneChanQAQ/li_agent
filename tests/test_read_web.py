"""网页阅读工具的行为测试；HTTP 响应由本地模拟，不访问真实网站。"""

import json
import unittest

import httpx
from jsonschema.exceptions import ValidationError

from harness.provider.models import ToolCall
from harness.tool.executor import ToolExecutor
from harness.tool.read_web import ReadWebTool


class TestReadWebTool(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.url = "https://example.com/article"
        self.requests = []
        self.status = 200
        self.payload = {
            "results": [{"url": self.url, "raw_content": "这是测试正文"}],
            "failed_results": [],
        }
        self.client = httpx.AsyncClient(
            transport=httpx.MockTransport(self.handle_request),
        )
        self.addAsyncCleanup(self.client.aclose)
        self.tool = ReadWebTool(self.client, "test-key")

    def handle_request(self, request):
        self.requests.append(request)
        return httpx.Response(self.status, json=self.payload)

    async def test_read_web_returns_content_and_sends_extract_request(self):
        # 准备在 setUp 中完成；这里调用真实工具，再检查结果。
        result = await self.tool.execute({"url": f" {self.url} "})
        self.assertEqual(json.loads(result), {
            "requested_url": self.url,
            "url": self.url,
            "content": "这是测试正文",
            "extracted_chars": 6,
            "returned_chars": 6,
            "truncated": False,
        })
        self.assertEqual(len(self.requests), 1)
        request = self.requests[0]
        self.assertEqual(request.method, "POST")
        self.assertEqual(str(request.url), "https://api.tavily.com/extract")
        self.assertEqual(request.headers["Authorization"], "Bearer test-key")
        body = json.loads(request.content)
        self.assertEqual(body["urls"], [self.url])
        self.assertEqual(body["format"], "markdown")
        self.assertEqual(body["extract_depth"], "basic")
        self.assertNotIn("query", body)

    async def test_truncation_below_at_and_above_character_limit(self):
        tool = ReadWebTool(self.client, "test-key", max_chars=3)
        for content in ("中文", "中文🙂", "中文🙂尾"):
            with self.subTest(content=content):
                self.payload["results"][0]["raw_content"] = content
                data = json.loads(await tool.execute({"url": self.url}))
                self.assertEqual(data["content"], content[:3])
                self.assertEqual(data["extracted_chars"], len(content))
                self.assertEqual(data["returned_chars"], min(len(content), 3))
                self.assertEqual(data["truncated"], len(content) > 3)

    async def test_requested_and_returned_urls_are_preserved_separately(self):
        self.payload["results"][0]["url"] = "https://example.com/canonical"
        data = json.loads(await self.tool.execute({"url": self.url}))
        self.assertEqual(data["requested_url"], self.url)
        self.assertEqual(data["url"], "https://example.com/canonical")

    async def test_blank_key_and_nonpositive_limit_are_rejected(self):
        for key in ("", " \n "):
            with self.subTest(key=key), self.assertRaises(ValueError):
                ReadWebTool(self.client, key)
        for limit in (0, -1):
            with self.subTest(limit=limit), self.assertRaises(ValueError):
                ReadWebTool(self.client, "test-key", max_chars=limit)
        self.assertEqual(self.requests, [])

    async def test_invalid_urls_are_rejected_before_http_request(self):
        for url in ("", " ", "example.com", "https:///missing-host",
                    "file:///tmp/test", "ftp://example.com/file"):
            with self.subTest(url=url), self.assertRaises(ValueError):
                await self.tool.execute({"url": url})
        self.assertEqual(self.requests, [])

    async def test_executor_rejects_invalid_arguments_before_http_request(self):
        executor = ToolExecutor([self.tool])
        for arguments in ({}, {"url": 42}, {"url": ""},
                          {"url": self.url, "unexpected": True}):
            with self.subTest(arguments=arguments), self.assertRaises(ValidationError):
                await executor.execute(
                    ToolCall(tool_call_id="read-1", tool_name="read_web",
                             tool_input=arguments),
                    allowed_tool_names=frozenset({"read_web"}),
                )
        self.assertEqual(self.requests, [])

    async def test_http_errors_raise_before_response_shape_validation(self):
        self.payload = {"detail": {"error": "service error"}}
        for status in (401, 429, 500):
            self.status = status
            with self.subTest(status=status), self.assertRaises(httpx.HTTPStatusError) as caught:
                await self.tool.execute({"url": self.url})
            self.assertEqual(caught.exception.response.status_code, status)

    async def test_network_errors_propagate_without_retry(self):
        for error_type in (httpx.ReadTimeout, httpx.ConnectError):
            with self.subTest(error_type=error_type):
                requests = []

                def fail(request):
                    requests.append(request)
                    raise error_type("模拟网络失败", request=request)

                async with httpx.AsyncClient(transport=httpx.MockTransport(fail)) as client:
                    tool = ReadWebTool(client, "test-key")
                    with self.assertRaises(error_type):
                        await tool.execute({"url": self.url})
                self.assertEqual(len(requests), 1)

    async def test_http_200_extraction_failure_is_not_treated_as_success(self):
        failure = {"url": self.url, "error": "cannot extract page"}
        for results in ([], self.payload["results"]):
            with self.subTest(results=results):
                self.payload = {"results": results, "failed_results": [failure]}
                with self.assertRaisesRegex(RuntimeError, "cannot extract page"):
                    await self.tool.execute({"url": self.url})

    async def test_malformed_response_shapes_are_rejected(self):
        cases = [
            [], {}, {"results": {}, "failed_results": []},
            {"results": [], "failed_results": None},
            {"results": [], "failed_results": ["invalid failure"]},
            {"results": [], "failed_results": []},
            {"results": [{}, {}], "failed_results": []},
            {"results": ["invalid result"], "failed_results": []},
        ]
        for payload in cases:
            with self.subTest(payload=payload), self.assertRaises(ValueError):
                self.payload = payload
                await self.tool.execute({"url": self.url})

    async def test_empty_or_nonstring_content_is_rejected(self):
        for content in (None, "", " \n ", 42):
            with self.subTest(content=content), self.assertRaises(RuntimeError):
                self.payload["results"][0]["raw_content"] = content
                await self.tool.execute({"url": self.url})

    async def test_missing_or_invalid_result_url_is_rejected(self):
        for url in (None, "", " ", 42):
            with self.subTest(url=url), self.assertRaises(ValueError):
                self.payload["results"][0]["url"] = url
                await self.tool.execute({"url": self.url})

    async def test_non_json_response_is_rejected(self):
        transport = httpx.MockTransport(lambda request: httpx.Response(200, text="not JSON"))
        async with httpx.AsyncClient(transport=transport) as client:
            with self.assertRaises(json.JSONDecodeError):
                await ReadWebTool(client, "test-key").execute({"url": self.url})


if __name__ == "__main__":
    unittest.main()
