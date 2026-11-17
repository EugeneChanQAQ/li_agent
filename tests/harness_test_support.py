"""测试替身只替换模型客户端；其余执行链使用真实实现。"""

from copy import deepcopy
from types import SimpleNamespace

from harness.provider.deepseek_provider import DeepSeekProvider
from harness.runtime.loop import AgentLoop
from harness.runtime.runtime import AgentRuntime
from harness.runtime.store import MemoryStore
from harness.tool.executor import ToolExecutor
from harness.tool.models import ToolDefinition


def model_response(content="完成", *, reason="stop", calls=None):
    return SimpleNamespace(
        choices=[SimpleNamespace(
            finish_reason=reason,
            message=SimpleNamespace(content=content, tool_calls=calls or []),
        )],
        usage=None,
    )


def tool_response(*, arguments='{"value":"证据"}', name="echo", call_id="call_1"):
    call = SimpleNamespace(
        id=call_id,
        type="function",
        function=SimpleNamespace(name=name, arguments=arguments),
    )
    return model_response(None, reason="tool_calls", calls=[call])


class ScriptedClient:
    """按顺序返回预设响应，保存发送到模型端的请求快照。"""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []
        self.chat = SimpleNamespace(completions=self)

    async def create(self, **kwargs):
        self.requests.append(deepcopy(kwargs))
        if not self.responses:
            raise AssertionError("发生了未预期的额外模型调用")
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class EchoTool:
    """确定性工具：不依赖时钟或网络，记录实际执行参数。"""

    def __init__(self, error=None):
        self.definition = ToolDefinition(
            name="echo",
            description="返回输入文本",
            input_schema={
                "type": "object",
                "properties": {"value": {"type": "string"}},
                "required": ["value"],
                "additionalProperties": False,
            },
        )
        self.calls = []
        self.error = error

    async def execute(self, arguments):
        self.calls.append(deepcopy(arguments))
        if self.error is not None:
            raise self.error
        return arguments["value"]


class RecordingStore(MemoryStore):
    def __init__(self, fail_on=None):
        super().__init__()
        self.saved = []
        self.fail_on = fail_on

    async def save(self, record):
        if len(self.saved) + 1 == self.fail_on:
            raise OSError("模拟存储失败")
        await super().save(record)
        self.saved.append(record.model_copy(deep=True))


def build_runtime(responses, *, tool=None, max_steps=5, store=None):
    client = ScriptedClient(responses)
    tool = tool if tool is not None else EchoTool()
    store = store if store is not None else RecordingStore()
    runtime = AgentRuntime(
        loop=AgentLoop(DeepSeekProvider(client), ToolExecutor([tool]), max_steps),
        store=store,
    )
    return runtime, client, tool, store
