from pathlib import Path
from tempfile import NamedTemporaryFile

from harness.session.models import SessionRecord


class JsonSessionStore:
    def __init__(self, path: Path) -> None:
        self._path = path

    def load(self) -> SessionRecord | None:
        try:
            content = self._path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None
        return SessionRecord.model_validate_json(content)


    def save(self, record: SessionRecord) -> None:
        # 先序列化，再操作文件
        content = record.model_dump_json(indent=2, ensure_ascii=False)

        self._path.parent.mkdir(parents=True, exist_ok=True)

        temp_path: Path | None = None

        try:
            # 临时文件与目标文件放在同一个目录。
            with NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self._path.parent,
                prefix=f".{self._path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temp_file:
                temp_path = Path(temp_file.name)
                temp_file.write(content)

            # 退出 with 后文件已关闭，再替换目标文件。
            temp_path.replace(self._path)

        finally:
            # 写入或替换失败时，尝试清理临时文件。
            # 替换成功后，临时路径已经不存在。
            if temp_path is not None:
                temp_path.unlink(missing_ok=True)