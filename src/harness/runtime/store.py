"""
保存和读取记录，不决定执行状态如何变化。
当前采用内存存储，用于验证与 Runtime 的协作。
后续可增加数据库存储实现。

保存 Run 轨迹
"""
from pathlib import Path
import re
from tempfile import NamedTemporaryFile

from harness.runtime.models import RunRecord

class MemoryStore:
    def __init__(self):
        self._records: dict[str, RunRecord] = {}

    async def save(self, record: RunRecord) -> None:
        """保存记录；相同 run_id 的记录会被覆盖。"""
        self._records[record.run_id] = record.model_copy(deep=True)

    async def get(self, run_id: str) -> RunRecord | None:
        """返回记录的独立副本，不存在时返回 None"""
        record = self._records.get(run_id)
        if record is None:
            return None
        return record.model_copy(deep=True)

class JsonRunStore:
    """将 RunRecord 保存为 JSON 文件，便于调试和分析。"""
    def __init__(self, directory: Path):
        self._directory = directory

    def _record_path(self, run_id: str) -> Path:
        # run_id 作为文件名，限制字符，避免路径越界
        if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
            raise ValueError("run_id 只能包含字母、数字、下划线和连字符。")
        return self._directory / f"{run_id}.json"

    async def save(self, record: RunRecord) -> None:
        path = self._record_path(record.run_id)

        # 先序列化，再操作文件
        content = record.model_dump_json(indent=2, ensure_ascii=False)
        self._directory.mkdir(parents=True, exist_ok=True)

        temp_path: Path | None = None

        try:
            with NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self._directory,
                prefix=f".{record.run_id}.",
                suffix=".tmp",
                delete=False,
            ) as temp_file:
                temp_path = Path(temp_file.name)
                temp_file.write(content)

            # 文件关闭后再替换；同一个 run_id 覆盖同一个文件。
            temp_path.replace(path)

        finally:
            # 写入或替换失败时，尝试清理临时文件。
            # 替换成功后，临时路径已经不存在。
            try:
                if temp_path is not None:
                    temp_path.unlink(missing_ok=True) # missing_ok=True 表示文件不存在也不报错，避免替换成功后再次删除报错
            except OSError:
                    # tips：清理失败不遮蔽原始写入异常。
                    pass


    async def get(self, run_id: str) -> RunRecord | None:
        path = self._record_path(run_id)

        try:
            content = path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return None

        # 损坏文件、读取权限错误等继续向外抛出。
        record = RunRecord.model_validate_json(content)

        if record.run_id != run_id:
            raise ValueError("文件中的 run_id 与查询的 run_id 不一致。")

        return record

            
                
