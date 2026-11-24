from datetime import datetime, timezone

from harness.provider.models import ToolDefinition, Message, ToolMessage, LLMRequest
from harness.runtime.runtime import AgentRuntime, RunRecord

from harness.session.store import JsonSessionStore
from harness.session.models import SessionRecord

class SessionSaveError(Exception):
    """保存会话记录失败时抛出此异常"""
    def __init__(self, run_record: RunRecord):
        super().__init__("本轮执行已完成，但会话历史保存失败。")
        self.run_record = run_record

class ChatSession:
    def __init__(
            self,
            runtime: AgentRuntime,
            model: str,
            tools: list[ToolDefinition],
            store: JsonSessionStore,
            record: SessionRecord,
            ) -> None:
        # 创建会话时保存配置，不发起请求。
        self._runtime = runtime
        self._model = model
        self._tools = [
            tool.model_copy(deep=True)
            for tool in tools
        ]
        self._history: list[Message | ToolMessage] = []

        self._store = store

        # 防止外部修改传入的记录影响当前会话
        self._record = record.model_copy(deep=True)
        

    async def send_message(self, text: str) -> RunRecord:
        # 发送消息调用runtime，进入runtime生命周期。并返回RunRecord，包含运行状态、结果和错误信息。
        text = text.strip()
        if not text:
            raise ValueError("消息不能为空。")

        # 复制旧历史，加入本轮输入
        # 此处尚未修改history

        messages = [
            message.model_copy(deep=True)
            for message in self._record.messages
        ]

        messages.append(
            Message(
                role="user",
                content=text,
            )
        )

        # 每次发消息构建本轮请求
        request = LLMRequest(
            model=self._model,
            messages=messages,
            tools=[
                tool.model_copy(deep=True)
                for tool in self._tools
            ],
        )

        run_record = await self._runtime.run(
            request,
            session_id=self._record.session_id,
        )

        # 正常完成才保存历史
        if run_record.status == "completed":
            assert run_record.result is not None
            try:
                self._save_history(run_record.result.messages)
            except Exception as exc:
                # 保留已完成的 Run，供入口展示回答。
                raise SessionSaveError(run_record) from exc

        return run_record

    def clear(self) -> None:
        self._save_history([])

    def _save_history(
        self,
        messages: list[Message | ToolMessage],
    ) -> None:
        # 构造新记录会重新执行模型校验。
        new_record = SessionRecord(
            session_id=self._record.session_id,
            created_at=self._record.created_at,
            updated_at=datetime.now(timezone.utc),
            messages=[
                message.model_copy(deep=True)
                for message in messages
            ],
        )

        # 保存失败时，下面的内存替换不会执行。
        self._store.save(new_record)
        self._record = new_record

