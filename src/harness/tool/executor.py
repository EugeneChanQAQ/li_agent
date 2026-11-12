# 把模型返回的 ToolCall 在 Loop 循环中进行执行，然后返回 ToolMessage

from typing import Any, Protocol

from jsonschema import validate

from harness.provider.models import ToolMessage, ToolCall
from harness.tool.models import ToolDefinition


class ExecutableTool(Protocol):
    # 在 ToolDefinition 外面多定义一层 definition，
    # 为了集中管理工具的描述信息，方便统一校验并提供给模型
    definition: ToolDefinition

    async def execute(self, arguments: dict[str, Any]) -> str:
        ...

class ToolExecutor:
    def __init__(self, tools: list[ExecutableTool]):
        self._tools: dict[str, ExecutableTool] = {}

        for tool in tools:
            name = tool.definition.name

            if name in self._tools:
                raise ValueError(f"工具名称重复：{name}")

            self._tools[name] = tool

    async def execute(self, call: ToolCall, *, allowed_tool_names: frozenset[str]) -> ToolMessage:
        if call.tool_name not in allowed_tool_names:
            raise PermissionError(f"本次执行不允许使用工具：{call.tool_name}")
        
        tool = self._tools.get(call.tool_name)

        if tool is None:
            raise ValueError(f"未注册的工具：{call.tool_name}")

        # 校验调用 tool 的参数类型和规格是否符合
        validate_tool_input(
            arguments=call.tool_input,
            schema=tool.definition.input_schema
        )

        content = await tool.execute(call.tool_input)
        # tips：直接传入tool_input不会检查参数是否符合input_schema
        # 需要另外写检查参数逻辑

        # 类型注解不会自动执行运行时检查
        if not isinstance(content, str):
            raise  TypeError(f"工具{call.tool_name}必须返回字符串")

        return ToolMessage(
            tool_name=call.tool_name,
            tool_call_id=call.tool_call_id,
            content=content,
        )

def validate_tool_input(arguments, schema) -> None:
    validate(instance=arguments, schema=schema)

        
