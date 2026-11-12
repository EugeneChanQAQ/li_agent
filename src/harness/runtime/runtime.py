from uuid import uuid4
from datetime import  datetime, timezone

from .models import RunRecord
from harness.provider.models import LLMRequest
from harness.runtime.store import MemoryStore
from harness.runtime.loop import AgentLoop

class AgentRuntime:
    def __init__(self, loop: AgentLoop, store: MemoryStore):
        self.loop = loop
        self.store = store

    async def run(self, request: LLMRequest) -> RunRecord:
        run_id = str(uuid4())
        start_time = datetime.now(timezone.utc)

        running_record = RunRecord(
            run_id=run_id,
            status="running",
            started_at=start_time,
        )
        await self.store.save(running_record)

        try:
            result = await self.loop.run(request)
        except Exception as exc:
            final_record = RunRecord(
                run_id=run_id,
                status="failed",
                started_at=start_time,
                finished_at=datetime.now(timezone.utc),
                error=f"{type(exc).__name__}: {exc}"
            )
        else:
            status = ("completed"
                      if result.stop_reason == "completed"
                      else "stopped"
                      )
            final_record = RunRecord(
                run_id=run_id,
                status=status,
                started_at=start_time,
                finished_at=datetime.now(timezone.utc),
                result=result,
            )

        await self.store.save(final_record)
        return final_record


