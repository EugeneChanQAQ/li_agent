import unittest
from datetime import datetime, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from pydantic import ValidationError

from harness.provider.models import Message
from harness.session.models import SessionRecord
from harness.session.store import JsonSessionStore


class TestJsonSessionStore(unittest.TestCase):
    def setUp(self):
        directory = TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / "data" / "session.json"
        self.store = JsonSessionStore(self.path)
        now = datetime.now(timezone.utc)
        self.record = SessionRecord(
            session_id="fixed", created_at=now, updated_at=now,
            messages=[Message(role="user", content="中文历史")],
        )

    def test_missing_file_then_save_and_reload(self):
        self.assertIsNone(self.store.load())
        self.assertFalse(self.path.exists())
        self.store.save(self.record)
        self.assertEqual(JsonSessionStore(self.path).load(), self.record)

    def test_invalid_file_is_rejected_without_modification(self):
        self.path.parent.mkdir()
        for content in ('{broken', '{}', '{"session_id": "fixed", "messages": []}'):
            with self.subTest(content=content):
                self.path.write_text(content, encoding="utf-8")
                with self.assertRaises(ValidationError):
                    self.store.load()
                self.assertEqual(self.path.read_text(encoding="utf-8"), content)

    def test_read_error_is_not_treated_as_missing_file(self):
        self.store.save(self.record)
        with patch.object(Path, "read_text", side_effect=PermissionError("读取被拒绝")):
            with self.assertRaises(PermissionError):
                self.store.load()

    def test_replace_failure_preserves_old_file_and_removes_temporary_file(self):
        self.store.save(self.record)
        before = self.path.read_bytes()
        replacement = self.record.model_copy(update={"messages": []})
        with patch.object(Path, "replace", side_effect=OSError("替换失败")):
            with self.assertRaisesRegex(OSError, "替换失败"):
                self.store.save(replacement)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])

    def test_partial_write_failure_preserves_old_file_and_cleans_up(self):
        from tempfile import NamedTemporaryFile

        self.store.save(self.record)
        before = self.path.read_bytes()

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

        with patch("harness.session.store.NamedTemporaryFile", FailingWrite):
            with self.assertRaisesRegex(OSError, "写入中断"):
                self.store.save(self.record)
        self.assertEqual(self.path.read_bytes(), before)
        self.assertEqual(list(self.path.parent.iterdir()), [self.path])


if __name__ == "__main__":
    unittest.main()
