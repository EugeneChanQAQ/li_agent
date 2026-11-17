import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from harness.session.chat_session import ChatSession, SessionSaveError
from harness.session.models import SessionRecord
from harness.session.store import JsonSessionStore
from harness_test_support import (
    RecordingStore, build_runtime, model_response, tool_response,
)


class TestChatSession(unittest.IsolatedAsyncioTestCase):
    def new_session_storage(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        path = Path(directory.name) / "session.json"
        now = datetime.now(timezone.utc)
        record = SessionRecord(session_id="test-session", created_at=now, updated_at=now)
        store = JsonSessionStore(path)
        store.save(record)
        return store, record, path

    def build_session(self, responses, *, session_store=None, record=None, **kwargs):
        runtime, client, tool, store = build_runtime(responses, **kwargs)
        if session_store is None:
            session_store, record, _ = self.new_session_storage()
        session = ChatSession(runtime, "test-model", [tool.definition], session_store, record)
        return session, client, tool, store

    async def test_followup_contains_prior_turn_once_and_creates_new_run(self):
        session, client, _, store = self.build_session([
            model_response("第一答"), model_response("第二答"), model_response("第三答"),
        ])
        first = await session.send_message("  第一问  ")
        second = await session.send_message("第二问")
        third = await session.send_message("第三问")
        self.assertEqual(len({first.run_id, second.run_id, third.run_id}), 3)
        self.assertEqual(client.requests[2]["messages"], [
            {"role": "user", "content": "第一问"},
            {"role": "assistant", "content": "第一答"},
            {"role": "user", "content": "第二问"},
            {"role": "assistant", "content": "第二答"},
            {"role": "user", "content": "第三问"},
        ])
        self.assertEqual(await store.get(first.run_id), first)
        self.assertEqual(third.status, "completed")

    async def test_tool_followup_preserves_call_and_result(self):
        session, client, tool, _ = self.build_session([
            tool_response(), model_response("已取得证据"), model_response("解释证据"),
        ])
        await session.send_message("取证")
        record = await session.send_message("解释刚才的结果")
        self.assertEqual(record.status, "completed")
        messages = client.requests[2]["messages"]
        self.assertEqual([m["role"] for m in messages],
                         ["user", "assistant", "tool", "assistant", "user"])
        self.assertEqual(messages[1]["tool_calls"][0]["id"], "call_1")
        self.assertEqual(messages[2], {
            "role": "tool", "content": "证据", "tool_call_id": "call_1",
        })
        self.assertEqual(tool.calls, [{"value": "证据"}])

    async def test_unsuccessful_turn_does_not_enter_next_request(self):
        cases = [
            (RuntimeError("模型失败"), "failed"),
            (model_response("未完成", reason="length"), "stopped"),
            (tool_response(), "stopped"),
        ]
        for response, expected_status in cases:
            with self.subTest(status=expected_status, response=response):
                session, client, _, _ = self.build_session([
                    model_response("已保存"), response, model_response("继续成功"),
                ], max_steps=1)
                await session.send_message("保留这轮")
                failed = await session.send_message("不保存这轮")
                self.assertEqual(failed.status, expected_status)
                result = await session.send_message("继续")
                self.assertEqual(result.status, "completed")
                self.assertEqual(client.requests[-1]["messages"], [
                    {"role": "user", "content": "保留这轮"},
                    {"role": "assistant", "content": "已保存"},
                    {"role": "user", "content": "继续"},
                ])

    async def test_storage_exception_preserves_history_and_allows_next_turn(self):
        # 第一轮两次保存成功；第二轮在最终保存时失败。
        store = RecordingStore(fail_on=4)
        session, client, _, _ = self.build_session([
            model_response("已保存"), model_response("未提交"), model_response("继续成功"),
        ], store=store)
        await session.send_message("保留这轮")
        with self.assertRaisesRegex(OSError, "模拟存储失败"):
            await session.send_message("不保存这轮")
        store.fail_on = None
        record = await session.send_message("继续")
        self.assertEqual(record.status, "completed")
        self.assertEqual(client.requests[-1]["messages"], [
            {"role": "user", "content": "保留这轮"},
            {"role": "assistant", "content": "已保存"},
            {"role": "user", "content": "继续"},
        ])

    async def test_mutating_returned_record_does_not_change_history(self):
        session, client, _, _ = self.build_session([
            tool_response(), model_response("原回答"), model_response(),
        ])
        record = await session.send_message("原问题")
        record.result.messages[0].content = "被篡改的问题"
        record.result.messages[1].tool_calls[0].tool_input["value"] = "被篡改的参数"
        record.result.messages[2].content = "被篡改的结果"
        await session.send_message("追问")
        messages = client.requests[-1]["messages"]
        self.assertEqual(messages[0]["content"], "原问题")
        self.assertIn("证据", messages[1]["tool_calls"][0]["function"]["arguments"])
        self.assertNotIn("被篡改", messages[1]["tool_calls"][0]["function"]["arguments"])
        self.assertEqual(messages[2]["content"], "证据")

    async def test_clear_removes_history_but_keeps_tool_configuration(self):
        session, client, _, _ = self.build_session([model_response(), model_response()])
        await session.send_message("旧话题")
        session.clear()
        session.clear()  # 清空空会话也应该安全。
        await session.send_message("新话题")
        self.assertEqual(client.requests[-1]["messages"], [
            {"role": "user", "content": "新话题"},
        ])
        self.assertEqual(client.requests[-1]["tools"], client.requests[0]["tools"])

    async def test_blank_input_does_not_start_a_run_or_change_history(self):
        session, client, _, store = self.build_session([model_response(), model_response()])
        await session.send_message("旧问题")
        for text in ("", "  ", "\n\t"):
            with self.subTest(text=text), self.assertRaisesRegex(ValueError, "消息不能为空"):
                await session.send_message(text)
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(len(store.saved), 2)
        await session.send_message("追问")
        self.assertEqual(len(client.requests[-1]["messages"]), 3)

    async def test_sessions_sharing_runtime_have_independent_history(self):
        runtime, client, tool, _ = build_runtime([
            model_response("A答"), model_response("B答"), model_response(),
        ])
        first_store, first_record, _ = self.new_session_storage()
        second_store, second_record, _ = self.new_session_storage()
        first = ChatSession(runtime, "test-model", [tool.definition], first_store, first_record)
        second = ChatSession(runtime, "test-model", [tool.definition], second_store, second_record)
        await first.send_message("A问")
        await second.send_message("B问")
        self.assertEqual(client.requests[1]["messages"], [{"role": "user", "content": "B问"}])
        await first.send_message("A追问")
        self.assertEqual([m["content"] for m in client.requests[2]["messages"]],
                         ["A问", "A答", "A追问"])

    async def test_reloaded_session_preserves_tool_history_and_identity(self):
        store, initial, path = self.new_session_storage()
        session, _, _, _ = self.build_session(
            [tool_response(), model_response("已取得证据")],
            session_store=store, record=initial,
        )
        await session.send_message("取证")
        # 新建存储、会话与 Runtime，仅通过磁盘恢复历史。
        reopened_store = JsonSessionStore(path)
        saved = reopened_store.load()
        self.assertEqual(saved.session_id, initial.session_id)
        self.assertEqual(saved.created_at, initial.created_at)
        self.assertGreaterEqual(saved.updated_at, initial.updated_at)
        resumed, client, _, _ = self.build_session(
            [model_response("解释证据")], session_store=reopened_store, record=saved,
        )
        await resumed.send_message("解释刚才的结果")
        messages = client.requests[0]["messages"]
        self.assertEqual([m["role"] for m in messages],
                         ["user", "assistant", "tool", "assistant", "user"])
        self.assertEqual(messages[1]["tool_calls"][0]["id"], "call_1")
        self.assertEqual(messages[2]["tool_call_id"], "call_1")
        self.assertEqual(messages[2]["content"], "证据")
        self.assertIn("证据", messages[1]["tool_calls"][0]["function"]["arguments"])
        self.assertEqual(len(reopened_store.load().messages), 6)

    async def test_clear_is_persisted_for_reloaded_session(self):
        store, initial, path = self.new_session_storage()
        session, _, _, _ = self.build_session(
            [model_response()], session_store=store, record=initial,
        )
        await session.send_message("旧话题")
        session.clear()
        saved = JsonSessionStore(path).load()
        self.assertEqual(saved.messages, [])
        self.assertEqual(saved.session_id, initial.session_id)
        resumed, client, _, _ = self.build_session(
            [model_response()], session_store=JsonSessionStore(path), record=saved,
        )
        await resumed.send_message("新话题")
        self.assertEqual(client.requests[0]["messages"], [{"role": "user", "content": "新话题"}])

    async def test_session_save_failure_keeps_history_and_completed_run(self):
        store, initial, path = self.new_session_storage()
        session, client, _, run_store = self.build_session(
            [model_response("已保存"), model_response("未保存的回答"), model_response()],
            session_store=store, record=initial,
        )
        await session.send_message("保留这轮")
        before = path.read_bytes()
        with patch.object(Path, "replace", side_effect=OSError("模拟替换失败")):
            with self.assertRaises(SessionSaveError) as caught:
                await session.send_message("不保存这轮")
        failed_save = caught.exception
        self.assertIsInstance(failed_save.__cause__, OSError)
        self.assertEqual(failed_save.run_record.status, "completed")
        self.assertEqual(failed_save.run_record.result.final_text, "未保存的回答")
        self.assertEqual(await run_store.get(failed_save.run_record.run_id), failed_save.run_record)
        self.assertEqual(path.read_bytes(), before)
        await session.send_message("继续")
        self.assertEqual(client.requests[-1]["messages"], [
            {"role": "user", "content": "保留这轮"},
            {"role": "assistant", "content": "已保存"},
            {"role": "user", "content": "继续"},
        ])

    async def test_clear_failure_preserves_disk_and_memory_history(self):
        store, initial, path = self.new_session_storage()
        session, client, _, _ = self.build_session(
            [model_response("旧回答"), model_response()], session_store=store, record=initial,
        )
        await session.send_message("旧问题")
        before = path.read_bytes()
        with patch.object(Path, "replace", side_effect=OSError("模拟替换失败")):
            with self.assertRaises(OSError):
                session.clear()
        self.assertEqual(path.read_bytes(), before)
        await session.send_message("继续")
        self.assertEqual(client.requests[-1]["messages"], [
            {"role": "user", "content": "旧问题"},
            {"role": "assistant", "content": "旧回答"},
            {"role": "user", "content": "继续"},
        ])

    async def test_failed_and_stopped_runs_leave_saved_record_unchanged(self):
        for response in (RuntimeError("模型失败"), model_response("截断", reason="length")):
            with self.subTest(response=response):
                store, initial, path = self.new_session_storage()
                session, _, _, _ = self.build_session(
                    [model_response(), response], session_store=store, record=initial,
                )
                await session.send_message("已保存")
                before = path.read_bytes()
                result = await session.send_message("未完成")
                self.assertIn(result.status, ("failed", "stopped"))
                self.assertEqual(path.read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
