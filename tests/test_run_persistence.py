"""Run 文件存储和执行链回归；只使用临时目录与模拟模型。"""

import asyncio
import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from unittest.mock import patch

from pydantic import ValidationError

from harness.provider.models import LLMRequest, Message
from harness.runtime.models import RunRecord
from harness.runtime.store import JsonRunStore
from harness.session.chat_session import ChatSession, SessionSaveError
from harness.session.models import SessionRecord
from harness.session.store import JsonSessionStore
from harness_test_support import build_runtime, model_response, tool_response


class TestJsonRunStore(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.directory = Path(directory.name) / "runs"
        self.store = JsonRunStore(self.directory)
        self.record = RunRecord(
            run_id="run_1", session_id="session-1", status="running",
            started_at=datetime.now(timezone.utc),
            request=LLMRequest(
                model="test-model", messages=[Message(role="user", content="中文任务")],
            ),
        )

    async def test_missing_record_returns_none_without_creating_directory(self):
        self.assertIsNone(await self.store.get("missing"))
        self.assertFalse(self.directory.exists())

    async def test_reopened_store_preserves_record_and_isolates_loaded_values(self):
        await self.store.save(self.record)
        expected = self.record.model_copy(deep=True)
        self.record.request.messages[0].content = "保存后修改"
        reopened = JsonRunStore(self.directory)
        loaded = await reopened.get("run_1")
        self.assertEqual(loaded, expected)
        loaded.request.messages[0].content = "读取后修改"
        self.assertEqual(await reopened.get("run_1"), expected)
        self.assertEqual(list(self.directory.iterdir()), [self.directory / "run_1.json"])

    async def test_overwriting_one_run_keeps_other_runs(self):
        await self.store.save(self.record)
        other = self.record.model_copy(update={"run_id": "run_2"}, deep=True)
        await self.store.save(other)
        failed = RunRecord(
            **{**self.record.model_dump(), "status": "failed",
               "finished_at": datetime.now(timezone.utc), "error": "模型失败"},
        )
        await self.store.save(failed)
        reopened = JsonRunStore(self.directory)
        self.assertEqual(await reopened.get("run_1"), failed)
        self.assertEqual(await reopened.get("run_2"), other)
        self.assertEqual(len(list(self.directory.iterdir())), 2)

    async def test_invalid_identifiers_are_rejected_before_file_access(self):
        for run_id in ("", " ", "../escape", "/absolute", "a/b", "a\\b", ".", ".."):
            with self.subTest(run_id=run_id):
                with self.assertRaises(ValueError):
                    await self.store.get(run_id)
                invalid = self.record.model_copy(update={"run_id": run_id})
                with self.assertRaises(ValueError):
                    await self.store.save(invalid)
        self.assertFalse(self.directory.exists())

    async def test_corrupt_or_invalid_record_is_rejected_without_overwrite(self):
        self.directory.mkdir()
        path = self.directory / "run_1.json"
        for content in ("{broken", "{}", self.record.model_dump_json().replace('"running"', '"unknown"')):
            with self.subTest(content=content):
                path.write_text(content, encoding="utf-8")
                with self.assertRaises(ValidationError):
                    await self.store.get("run_1")
                self.assertEqual(path.read_text(encoding="utf-8"), content)

    async def test_record_id_must_match_queried_filename(self):
        await self.store.save(self.record)
        path = self.directory / "run_1.json"
        path.rename(self.directory / "different.json")
        with self.assertRaisesRegex(ValueError, "run_id"):
            await self.store.get("different")

    async def test_read_permission_error_propagates(self):
        await self.store.save(self.record)
        with patch.object(Path, "read_text", side_effect=PermissionError("读取被拒绝")):
            with self.assertRaises(PermissionError):
                await self.store.get("run_1")

    async def test_replace_failure_preserves_old_record_and_cleans_temporary_file(self):
        await self.store.save(self.record)
        path = self.directory / "run_1.json"
        before = path.read_bytes()
        with patch.object(Path, "replace", side_effect=OSError("替换失败")):
            with self.assertRaisesRegex(OSError, "替换失败"):
                await self.store.save(self.record)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(list(self.directory.iterdir()), [path])

    async def test_partial_write_failure_preserves_old_record_and_cleans_temporary_file(self):
        await self.store.save(self.record)
        path = self.directory / "run_1.json"
        before = path.read_bytes()

        class FailingWrite:
            def __init__(self, **kwargs):
                self.file = NamedTemporaryFile(**kwargs)

            def __enter__(self):
                self.file.__enter__()
                self.name = self.file.name
                return self

            def write(self, content):
                self.file.write(content[:10])
                raise OSError("写入中断")

            def __exit__(self, *args):
                return self.file.__exit__(*args)

        with patch("harness.runtime.store.NamedTemporaryFile", FailingWrite):
            with self.assertRaisesRegex(OSError, "写入中断"):
                await self.store.save(self.record)
        self.assertEqual(path.read_bytes(), before)
        self.assertEqual(list(self.directory.iterdir()), [path])

    async def test_cleanup_error_does_not_mask_original_save_error(self):
        await self.store.save(self.record)
        before = (self.directory / "run_1.json").read_bytes()
        with patch.object(Path, "replace", side_effect=OSError("原始替换错误")), \
                patch.object(Path, "unlink", side_effect=PermissionError("清理失败")):
            with self.assertRaisesRegex(OSError, "原始替换错误"):
                await self.store.save(self.record)
        self.assertEqual((self.directory / "run_1.json").read_bytes(), before)


class TestRunPersistenceIntegration(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.runs = self.root / "runs"

    async def test_reopened_store_preserves_all_outcomes_and_tool_history(self):
        cases = [
            ([tool_response(), model_response("完成")], 5, "completed", "completed"),
            ([tool_response()], 1, "stopped", "max_steps"),
            ([model_response("半句话", reason="length")], 5, "stopped", "output_truncated"),
            ([RuntimeError("模型不可用")], 5, "failed", None),
        ]
        for responses, max_steps, status, reason in cases:
            with self.subTest(status=status, reason=reason):
                runtime, _, tool, _ = build_runtime(
                    responses, max_steps=max_steps, store=JsonRunStore(self.runs),
                )
                request = LLMRequest(
                    model="test-model", messages=[Message(role="user", content="执行任务")],
                    tools=[tool.definition],
                )
                record = await runtime.run(request, session_id="persisted-session")
                saved = await JsonRunStore(self.runs).get(record.run_id)
                self.assertEqual(saved, record)
                self.assertEqual(saved.status, status)
                self.assertEqual(saved.session_id, "persisted-session")
                self.assertEqual(saved.request, request)
                if reason is not None:
                    self.assertEqual(saved.result.stop_reason, reason)
                else:
                    self.assertIn("模型不可用", saved.error)
                    self.assertIsNone(saved.result)
                if reason in ("completed", "max_steps"):
                    call = saved.result.messages[1].tool_calls[0]
                    result = saved.result.messages[2]
                    self.assertEqual(call.tool_input, {"value": "证据"})
                    self.assertEqual(result.tool_call_id, call.tool_call_id)
                    self.assertEqual(result.tool_name, "echo")
                    self.assertEqual(result.content, "证据")
        self.assertEqual(len(list(self.runs.glob("*.json"))), 4)

    def build_session(self, responses):
        runtime, client, tool, _ = build_runtime(responses, store=JsonRunStore(self.runs))
        now = datetime.now(timezone.utc)
        initial = SessionRecord(session_id="disk-session", created_at=now, updated_at=now)
        session_store = JsonSessionStore(self.root / "session.json")
        session_store.save(initial)
        return ChatSession(runtime, "test-model", [tool.definition], session_store, initial), client, session_store

    async def test_session_and_runs_remain_linked_after_reloading(self):
        session, _, session_store = self.build_session([tool_response(), model_response("已取证")])
        first = await session.send_message("取证")
        loaded_session = session_store.load()
        runtime, client, tool, _ = build_runtime([model_response("解释证据")], store=JsonRunStore(self.runs))
        reopened = ChatSession(runtime, "test-model", [tool.definition], session_store, loaded_session)
        second = await reopened.send_message("继续")
        self.assertNotEqual(first.run_id, second.run_id)
        for record in (first, second):
            saved = await JsonRunStore(self.runs).get(record.run_id)
            self.assertEqual(saved, record)
            self.assertEqual(saved.session_id, loaded_session.session_id)
        self.assertEqual(second.request.messages[:-1], first.result.messages)
        self.assertEqual(client.requests[0]["messages"][2]["tool_call_id"], "call_1")
        reopened.clear()
        self.assertEqual(session_store.load().messages, [])
        self.assertEqual(await JsonRunStore(self.runs).get(first.run_id), first)

    async def test_initial_disk_failure_prevents_model_call_and_history_update(self):
        session, client, session_store = self.build_session([model_response()])
        before = session_store.load()
        with patch("harness.runtime.store.NamedTemporaryFile", side_effect=PermissionError("禁止写入")):
            with self.assertRaises(PermissionError):
                await session.send_message("不应调用模型")
        self.assertEqual(client.requests, [])
        self.assertEqual(session_store.load(), before)
        self.assertEqual(list(self.runs.iterdir()), [])

    async def test_final_disk_failure_keeps_running_record_and_old_session(self):
        session, client, session_store = self.build_session([model_response("未落盘的回答")])
        before = session_store.load()
        original_replace = Path.replace
        run_replacements = 0

        def fail_final_run_replace(source, target):
            nonlocal run_replacements
            if Path(target).parent == self.runs:
                run_replacements += 1
                if run_replacements == 2:
                    raise OSError("结束记录保存失败")
            return original_replace(source, target)

        with patch.object(Path, "replace", fail_final_run_replace):
            with self.assertRaisesRegex(OSError, "结束记录保存失败"):
                await session.send_message("执行任务")
        self.assertEqual(len(client.requests), 1)
        self.assertEqual(session_store.load(), before)
        files = list(self.runs.iterdir())
        self.assertEqual(len(files), 1)
        saved = await JsonRunStore(self.runs).get(files[0].stem)
        self.assertEqual(saved.status, "running")
        self.assertIsNone(saved.finished_at)
        self.assertIsNone(saved.result)
        self.assertEqual(saved.request.messages[0].content, "执行任务")

    async def test_session_save_failure_leaves_completed_run_on_disk(self):
        session, _, session_store = self.build_session([model_response("已完成")])
        before = session_store.load()
        with patch.object(session_store, "save", side_effect=OSError("会话保存失败")):
            with self.assertRaises(SessionSaveError) as caught:
                await session.send_message("执行任务")
        record = caught.exception.run_record
        self.assertEqual(record.status, "completed")
        self.assertEqual(await JsonRunStore(self.runs).get(record.run_id), record)
        self.assertEqual(session_store.load(), before)

    async def test_cancelled_execution_leaves_readable_running_record(self):
        runtime, client, tool, _ = build_runtime([], store=JsonRunStore(self.runs))
        entered = asyncio.Event()

        async def wait_for_cancellation(**kwargs):
            entered.set()
            await asyncio.Future()

        client.create = wait_for_cancellation
        request = LLMRequest(model="test-model", messages=[Message(role="user", content="执行任务")])
        task = asyncio.create_task(runtime.run(request, session_id="cancelled-session"))
        try:
            await asyncio.wait_for(entered.wait(), timeout=2)
        finally:
            task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await task
        files = list(self.runs.iterdir())
        self.assertEqual(len(files), 1)
        saved = await JsonRunStore(self.runs).get(files[0].stem)
        self.assertEqual(saved.status, "running")
        self.assertEqual(saved.session_id, "cancelled-session")
        self.assertIsNone(saved.finished_at)
        self.assertEqual(saved.request, request)


if __name__ == "__main__":
    unittest.main()
