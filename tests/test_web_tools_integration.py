"""使用模拟模型和 HTTP 响应，验证真实的搜索、阅读与 Runtime 执行链。"""

import json
import unittest

import httpx

from harness.provider.deepseek_provider import DeepSeekProvider
from harness.provider.models import LLMRequest, Message
from harness.runtime.loop import AgentLoop
from harness.runtime.runtime import AgentRuntime
from harness.tool.executor import ToolExecutor
from harness.tool.read_web import ReadWebTool
from harness.tool.web_search import WebSearchTool
from harness_test_support import RecordingStore, ScriptedClient, model_response, tool_response


class TestWebToolsIntegration(unittest.IsolatedAsyncioTestCase):
    async def test_search_then_read_returns_evidence_to_model_and_saves_run(self):
        url = "https://example.com/article"
        requests = []

        def handle(request):
            requests.append(request)
            if request.url.path == "/search":
                return httpx.Response(200, json={"results": [
                    {"title": "官方资料", "url": url, "content": "搜索摘要"},
                ]})
            self.assertEqual(request.url.path, "/extract")
            return httpx.Response(200, json={
                "results": [{"url": url, "raw_content": "网页证据正文"}],
                "failed_results": [],
            })

        model = ScriptedClient([
            tool_response(name="web_search", arguments='{"query":"资料"}', call_id="search-1"),
            tool_response(name="read_web", arguments=json.dumps({"url": url}), call_id="read-1"),
            model_response("依据网页证据完成"),
        ])
        store = RecordingStore()
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            tools = [WebSearchTool(client, "test-key"), ReadWebTool(client, "test-key")]
            runtime = AgentRuntime(
                AgentLoop(DeepSeekProvider(model), ToolExecutor(tools), max_steps=3), store,
            )
            record = await runtime.run(LLMRequest(
                model="test-model", messages=[Message(role="user", content="搜索并阅读资料")],
                tools=[tool.definition for tool in tools],
            ), session_id="web-session")

        self.assertEqual(record.status, "completed")
        self.assertEqual(record.result.final_text, "依据网页证据完成")
        self.assertEqual([request.url.path for request in requests], ["/search", "/extract"])
        self.assertEqual(json.loads(requests[1].content)["urls"], [url])
        self.assertEqual(len(model.requests), 3)
        self.assertEqual(
            {tool["function"]["name"] for tool in model.requests[0]["tools"]},
            {"web_search", "read_web"},
        )
        search_message = model.requests[1]["messages"][-1]
        self.assertEqual(search_message["role"], "tool")
        self.assertEqual(search_message["tool_call_id"], "search-1")
        self.assertEqual(json.loads(search_message["content"])["results"][0]["url"], url)
        read_message = model.requests[2]["messages"][-1]
        self.assertEqual(read_message["tool_call_id"], "read-1")
        self.assertEqual(json.loads(read_message["content"])["content"], "网页证据正文")
        self.assertEqual([saved.status for saved in store.saved], ["running", "completed"])
        self.assertEqual(await store.get(record.run_id), record)

    async def test_tool_failures_are_saved_without_another_model_call(self):
        cases = [
            (WebSearchTool, "web_search", {"query": "资料"}, 429,
             {"detail": {"error": "rate limited"}}, "HTTPStatusError"),
            (ReadWebTool, "read_web", {"url": "https://example.com"}, 200,
             {"results": [], "failed_results": [{"error": "cannot extract"}]}, "RuntimeError"),
        ]
        for tool_class, name, arguments, status, payload, error_type in cases:
            with self.subTest(tool=name):
                requests = []

                def handle(request):
                    requests.append(request)
                    return httpx.Response(status, json=payload)

                model = ScriptedClient([
                    tool_response(name=name, arguments=json.dumps(arguments)),
                ])
                store = RecordingStore()
                async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
                    tool = tool_class(client, "test-key")
                    runtime = AgentRuntime(
                        AgentLoop(DeepSeekProvider(model), ToolExecutor([tool])), store,
                    )
                    record = await runtime.run(LLMRequest(
                        model="test-model", messages=[Message(role="user", content="获取资料")],
                        tools=[tool.definition],
                    ), session_id="failure-session")

                self.assertEqual(record.status, "failed")
                self.assertIn(error_type, record.error)
                self.assertIsNone(record.result)
                self.assertEqual(len(requests), 1)
                self.assertEqual(len(model.requests), 1)
                self.assertEqual([saved.status for saved in store.saved], ["running", "failed"])
                self.assertEqual(await store.get(record.run_id), record)


if __name__ == "__main__":
    unittest.main()
