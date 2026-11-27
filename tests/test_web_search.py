"""网页搜索工具的行为测试；使用模拟响应，不需要真实 API 密钥。"""

import json
import unittest

import httpx
from jsonschema.exceptions import ValidationError

from harness.provider.models import ToolCall
from harness.tool.executor import ToolExecutor
from harness.tool.web_search import WebSearchTool


class TestWebSearchTool(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.requests = []
        self.status = 200
        self.payload = {"results": [{
            "title": "测试标题", "url": "https://example.com/article",
            "content": "搜索摘要", "score": 0.9, "raw_content": None,
        }]}
        self.client = httpx.AsyncClient(
            transport=httpx.MockTransport(self.handle_request),
        )
        self.addAsyncCleanup(self.client.aclose)
        self.tool = WebSearchTool(self.client, "test-key")

    def handle_request(self, request):
        self.requests.append(request)
        return httpx.Response(self.status, json=self.payload)

    async def test_search_maps_results_and_sends_basic_search_request(self):
        result = await self.tool.execute({"query": "测试查询 "})
        self.assertEqual(json.loads(result), {
            "query": "测试查询",
            "results": [{"title": "测试标题", "url": "https://example.com/article",
                         "snippet": "搜索摘要"}],
        })
        self.assertEqual(len(self.requests), 1)
        request = self.requests[0]
        self.assertEqual(request.method, "POST")
        self.assertEqual(str(request.url), "https://api.tavily.com/search")
        self.assertEqual(request.headers["Authorization"], "Bearer test-key")
        body = json.loads(request.content)
        self.assertEqual(body["query"], "测试查询")
        self.assertEqual(body["max_results"], 5)
        self.assertEqual(body["search_depth"], "basic")
        self.assertFalse(body["include_answer"])
        self.assertFalse(body["include_raw_content"])
        self.assertFalse(body["auto_parameters"])

    async def test_explicit_result_limit_is_sent_and_applied(self):
        self.payload["results"] = [
            {"title": f"结果{i}", "url": f"https://example.com/{i}", "content": "摘要"}
            for i in range(3)
        ]
        result = json.loads(await self.tool.execute({"query": "测试", "max_results": 2}))
        self.assertEqual(json.loads(self.requests[0].content)["max_results"], 2)
        self.assertEqual([item["title"] for item in result["results"]], ["结果0", "结果1"])

    async def test_empty_results_are_a_successful_search(self):
        self.payload = {"results": []}
        self.assertEqual(json.loads(await self.tool.execute({"query": "测试"})),
                         {"query": "测试", "results": []})

    async def test_blank_key_and_blank_query_are_rejected(self):
        for value in ("", " \n "):
            with self.subTest(value=value):
                with self.assertRaises(ValueError):
                    WebSearchTool(self.client, value)
                with self.assertRaises(ValueError):
                    await self.tool.execute({"query": value})
        self.assertEqual(self.requests, [])

    async def test_executor_rejects_invalid_arguments_before_http_request(self):
        executor = ToolExecutor([self.tool])
        cases = [
            {}, {"query": ""}, {"query": " "}, {"query": 42},
            {"query": "测试", "max_results": 0},
            {"query": "测试", "max_results": 11},
            {"query": "测试", "max_results": "5"},
            {"query": "测试", "unexpected": True},
        ]
        for arguments in cases:
            with self.subTest(arguments=arguments), self.assertRaises(ValidationError):
                await executor.execute(
                    ToolCall(tool_call_id="search-1", tool_name="web_search",
                             tool_input=arguments),
                    allowed_tool_names=frozenset({"web_search"}),
                )
        self.assertEqual(self.requests, [])

    async def test_http_errors_raise_before_response_shape_validation(self):
        self.payload = {"detail": {"error": "service error"}}
        for status in (401, 429, 500):
            self.status = status
            with self.subTest(status=status), self.assertRaises(httpx.HTTPStatusError) as caught:
                await self.tool.execute({"query": "测试"})
            self.assertEqual(caught.exception.response.status_code, status)

    async def test_network_errors_propagate_without_retry(self):
        for error_type in (httpx.ReadTimeout, httpx.ConnectError):
            with self.subTest(error_type=error_type):
                requests = []

                def fail(request):
                    requests.append(request)
                    raise error_type("模拟网络失败", request=request)

                async with httpx.AsyncClient(transport=httpx.MockTransport(fail)) as client:
                    with self.assertRaises(error_type):
                        await WebSearchTool(client, "test-key").execute({"query": "测试"})
                self.assertEqual(len(requests), 1)

    async def test_non_list_results_are_rejected(self):
        for results in (None, {}, "invalid"):
            with self.subTest(results=results), self.assertRaises(ValueError):
                self.payload = {"results": results}
                await self.tool.execute({"query": "测试"})

    async def test_missing_required_response_fields_do_not_become_empty_success(self):
        # 当前实现用必填字段直接索引；字段缺失会明确抛错。
        for payload in ({}, {"results": [{}]}, {"results": [{"title": "标题"}]}):
            with self.subTest(payload=payload), self.assertRaises(KeyError):
                self.payload = payload
                await self.tool.execute({"query": "测试"})

    async def test_non_json_response_is_rejected(self):
        transport = httpx.MockTransport(lambda request: httpx.Response(200, text="not JSON"))
        async with httpx.AsyncClient(transport=transport) as client:
            with self.assertRaises(json.JSONDecodeError):
                await WebSearchTool(client, "test-key").execute({"query": "测试"})


if __name__ == "__main__":
    unittest.main()
