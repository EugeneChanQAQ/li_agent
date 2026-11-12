"""
保存和读取记录，不决定执行状态如何变化。
当前采用内存存储，用于验证与 Runtime 的协作。
后续可增加数据库存储实现。
"""

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
