import unittest

from harness.provider.models import LLMRequest, Message
from harness_test_support import (
    EchoTool, RecordingStore, build_runtime, model_response, tool_response,
)


class TestRuntimeIntegration(unittest.IsolatedAsyncioTestCase):
    def request(self, tool, *, allow_tool=True):
        return LLMRequest(
            model="test-model",
            messages=[Message(role="user", content="执行任务")],
            tools=[tool.definition] if allow_tool else [],
        )

    async def test_direct_answer_saves_running_and_completed(self):
        runtime, client, tool, store = build_runtime([model_response("你好")])
        record = await runtime.run(self.request(tool), session_id="test-session")
        self.assertEqual(record.status, "completed")
        self.assertEqual(record.result.final_text, "你好")
        self.assertEqual([r.status for r in store.saved], ["running", "completed"])
        self.assertTrue(all(r.run_id == record.run_id for r in store.saved))
        self.assertEqual(await store.get(record.run_id), record)
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(tool.calls, [])

    async def test_tool_result_reaches_model_with_matching_call_id(self):
        runtime, client, tool, store = build_runtime([
            tool_response(), model_response("依据证据完成"),
        ])
        request = self.request(tool)
        original = request.model_copy(deep=True)
        record = await runtime.run(request, session_id="test-session")
        self.assertEqual(record.status, "completed")
        self.assertEqual(record.result.final_text, "依据证据完成")
        self.assertEqual(tool.calls, [{"value": "证据"}])
        self.assertEqual(len(client.requests), 2)
        messages = client.requests[1]["messages"]
        self.assertEqual([m["role"] for m in messages], ["user", "assistant", "tool"])
        self.assertEqual(messages[1]["tool_calls"][0]["id"], "call_1")
        self.assertEqual(messages[2], {
            "role": "tool", "content": "证据", "tool_call_id": "call_1",
        })
        self.assertEqual(client.requests[0]["tools"][0]["function"]["name"], "echo")
        self.assertEqual(request, original)
        self.assertEqual(await store.get(record.run_id), record)

    async def test_max_steps_stops_without_an_extra_model_call(self):
        runtime, client, tool, _ = build_runtime([tool_response()], max_steps=1)
        record = await runtime.run(self.request(tool), session_id="test-session")
        self.assertEqual(record.status, "stopped")
        self.assertEqual(record.result.stop_reason, "max_steps")
        self.assertIsNone(record.result.final_text)
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(tool.calls, [{"value": "证据"}])
        self.assertEqual(record.result.messages[-1].role, "tool")

    async def test_truncated_response_is_not_added_to_history(self):
        runtime, _, tool, _ = build_runtime([model_response("半句话", reason="length")])
        record = await runtime.run(self.request(tool), session_id="test-session")
        self.assertEqual(record.status, "stopped")
        self.assertEqual(record.result.stop_reason, "output_truncated")
        self.assertEqual(len(record.result.messages), 1)
        self.assertIsNone(record.result.final_text)

    async def test_provider_exception_is_saved_as_failed(self):
        runtime, _, tool, store = build_runtime([RuntimeError("模型不可用")])
        record = await runtime.run(self.request(tool), session_id="test-session")
        self.assertEqual(record.status, "failed")
        self.assertIn("模型不可用", record.error)
        self.assertIsNone(record.result)
        self.assertEqual(await store.get(record.run_id), record)

    async def test_disallowed_tool_is_not_executed(self):
        runtime, _, tool, _ = build_runtime([tool_response()])
        record = await runtime.run(self.request(tool, allow_tool=False), session_id="test-session")
        self.assertEqual(record.status, "failed")
        self.assertIn("PermissionError", record.error)
        self.assertEqual(tool.calls, [])

    async def test_invalid_tool_arguments_are_rejected_before_execution(self):
        runtime, _, tool, _ = build_runtime([tool_response(arguments='{"value":42}')])
        record = await runtime.run(self.request(tool), session_id="test-session")
        self.assertEqual(record.status, "failed")
        self.assertIn("ValidationError", record.error)
        self.assertEqual(tool.calls, [])

    async def test_tool_exception_is_saved_as_failed(self):
        tool = EchoTool(error=RuntimeError("工具不可用"))
        runtime, client, _, store = build_runtime([tool_response()], tool=tool)
        record = await runtime.run(self.request(tool), session_id="test-session")
        self.assertEqual(record.status, "failed")
        self.assertIn("工具不可用", record.error)
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(await store.get(record.run_id), record)

    async def test_all_outcomes_preserve_session_and_independent_request_snapshot(self):
        cases = [
            ([tool_response(), model_response("完成")], 5, "completed"),
            ([tool_response()], 1, "stopped"),
            ([model_response("半句话", reason="length")], 5, "stopped"),
            ([RuntimeError("模型不可用")], 5, "failed"),
        ]
        for responses, max_steps, status in cases:
            with self.subTest(status=status, responses=responses):
                runtime, _, tool, store = build_runtime(responses, max_steps=max_steps)
                request = self.request(tool)
                original = request.model_copy(deep=True)
                record = await runtime.run(request, session_id="snapshot-session")

                self.assertEqual(record.status, status)
                self.assertEqual([r.status for r in store.saved], ["running", status])
                for saved in store.saved:
                    self.assertEqual(saved.session_id, "snapshot-session")
                    self.assertEqual(saved.request, original)

                # 修改调用方持有的嵌套数据，不应改变返回记录或已保存记录。
                request.messages[0].content = "后来修改的问题"
                request.messages.append(Message(role="user", content="后来追加的问题"))
                request.tools[0].input_schema["properties"]["value"]["type"] = "integer"

                self.assertEqual(record.session_id, "snapshot-session")
                self.assertEqual(record.request, original)
                persisted = await store.get(record.run_id)
                self.assertEqual(persisted.request, original)
                for saved in store.saved:
                    self.assertEqual(saved.request, original)

    async def test_initial_store_failure_prevents_model_call(self):
        runtime, client, tool, _ = build_runtime([], store=RecordingStore(fail_on=1))
        with self.assertRaisesRegex(OSError, "模拟存储失败"):
            await runtime.run(self.request(tool), session_id="test-session")
        self.assertEqual(client.requests, [])

    async def test_final_store_failure_propagates(self):
        store = RecordingStore(fail_on=2)
        runtime, client, tool, _ = build_runtime([model_response()], store=store)
        with self.assertRaisesRegex(OSError, "模拟存储失败"):
            await runtime.run(self.request(tool), session_id="test-session")
        self.assertEqual(len(client.requests), 1)
        saved = await store.get(store.saved[0].run_id)
        self.assertEqual(saved.status, "running")


if __name__ == "__main__":
    unittest.main()
