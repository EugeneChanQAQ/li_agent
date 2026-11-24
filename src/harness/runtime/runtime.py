from uuid import uuid4
from datetime import  datetime, timezone

from .models import RunRecord
from harness.provider.models import LLMRequest
from harness.runtime.store import MemoryStore, JsonRunStore
from harness.runtime.loop import AgentLoop

class AgentRuntime:
    def __init__(self, loop: AgentLoop, store: JsonRunStore | MemoryStore) -> None:
        self.loop = loop
        self.store = store

    async def run(self, request: LLMRequest, *, session_id: str) -> RunRecord:
        run_id = str(uuid4())
        start_time = datetime.now(timezone.utc)
        request_snapshot = request.model_copy(deep=True)

        running_record = RunRecord(
            run_id=run_id,
            session_id=session_id,
            request=request_snapshot,
            status="running",
            started_at=start_time,
        )
        await self.store.save(running_record)

        try:
            result = await self.loop.run(request)
        except Exception as exc:
            final_record = RunRecord(
                run_id=run_id,
                session_id=session_id,
                request=request_snapshot,
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
                session_id=session_id,
                request=request_snapshot,
                status=status,
                started_at=start_time,
                finished_at=datetime.now(timezone.utc),
                result=result,
            )

        await self.store.save(final_record)
        return final_record


