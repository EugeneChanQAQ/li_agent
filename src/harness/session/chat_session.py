from harness.provider.models import ToolDefinition, Message, ToolMessage, LLMRequest
from harness.runtime.runtime import AgentRuntime, RunRecord

class ChatSession:
    def __init__(
            self,
            runtime: AgentRuntime,
            model: str,
            tools: list[ToolDefinition],
            ) -> None:
        # 创建会话时保存配置，不发起请求。
        self._runtime = runtime
        self._model = model
        self._tools = [
            tool.model_copy(deep=True)
            for tool in tools
        ]
        self._history: list[Message | ToolMessage] = []

    async def send_message(self, text: str) -> RunRecord:
        # 发送消息调用runtime，进入runtime生命周期。并返回RunRecord，包含运行状态、结果和错误信息。
        text = text.strip()
        if not text:
            raise ValueError("消息不能为空。")

        # 复制旧历史，加入本轮输入
        # 此处尚未修改history

        messages = [
            message.model_copy(deep=True)
            for message in self._history
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

        record = await self._runtime.run(request)

        # 正常完成才保存历史
        if record.status == "completed":
            assert record.result is not None
            self._history = [
                message.model_copy(deep=True)
                for message in record.result.messages
            ]

        return record

    def clear(self) -> None:
        self._history.clear()

