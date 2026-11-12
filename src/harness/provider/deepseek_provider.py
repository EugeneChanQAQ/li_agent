import os
import json

from typing import Any

from openai import AsyncOpenAI
from harness.provider.models import LLMRequest, LLMResponse, ToolMessage, ToolCall
from harness.tool.models import ToolDefinition

# client应该避免顶层实例化，避免在导入时就进行网络请求。可以在需要使用时再创建实例，或者通过依赖注入的方式传入。
# client = AsyncOpenAI(
#     api_key=os.environ.get("deepseek_api_key"),
#     base_url=os.environ.get("deepseek_api_base_url", "https://api.deepseek.ai/v1"),
# )

class DeepSeekProvider:
    def __init__(self, client: AsyncOpenAI):
        self.client = client

    async def generate(self, request: LLMRequest) -> LLMResponse:
        # 将 LLMRequest 实例转换成deepseek API 所需要的格式
        payload = self._build_request(request)
        raw_response = await self.client.chat.completions.create(**payload)
        # 将 deepseek API 的响应转换为 LLMResponse 实例
        return self._parse_response(raw_response)


    def _build_request(self, request: LLMRequest) -> dict[str, Any]:
        # 将 LLMRequest 转换为 deepseek API 所需的请求格式
        # 针对本 harness 框架v1.0.0，主要做 message 和 tool 的 deepseek 格式转换
        messages: list[dict[str, Any]] = []
        for msg in request.messages:
            # 1. 工具执行消息结果
            if isinstance(msg, ToolMessage):
                if msg.content is None:
                    raise ValueError("工具消息结果不能为None")

                messages.append({
                    "role": "tool",
                    "content": msg.content,
                    "tool_call_id": msg.tool_call_id,
                })
                continue

            # 2. 普通消息或包含工具调用的 assistant 消息
            if msg.role in ("system", "user") and msg.content is None:
                raise ValueError(f"{msg.role} 消息的 content 不能为 None")

            converted: dict[str, Any] = {
                "role": msg.role,
                "content": msg.content,
            }
            
            if msg.tool_calls:
                converted["tool_calls"] = [
                    {
                        "id": call.tool_call_id,
                        "type": "function",
                        "function": {
                            "name": call.tool_name,
                            "arguments": json.dumps(
                                call.tool_input,
                                ensure_ascii=False,
                                allow_nan=False,
                            )
                        },
                    }
                    for call in msg.tool_calls
                ]
            messages.append(converted)

        # 3. 组装请求参数
        payload = {
            "messages": messages,
            "model": request.model,
            "stream": False,
        }

        if request.tools:
            payload["tools"] = [{
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.input_schema,
                },
            }
            for tool in request.tools
            ]
        return payload

    def _parse_response(self, raw_response) -> LLMResponse:
        # 将 deepseek API 的响应转换成 LLMResponse 实例
        
        # 1. 取出模型回复
        if not raw_response.choices:
            raise ValueError("模型中没有choices字段，无法获取模型回复")

        choice = raw_response.choices[0]
        message = choice.message

        if choice.finish_reason == "length":
            return LLMResponse(
                content=None,
                tool_calls=[],
                finish_reason="length",
                usage=(raw_response.usage.model_dump() if raw_response.usage is not None else None),
            )

        # 2. 转换模型提出的工具调用
        tool_calls: list[ToolCall] = []
        for call in message.tool_calls or []:
            if call.type != "function":
                raise ValueError(f"不支持的工具调用类型: {call.type}")

            try:
                tool_input = json.loads(call.function.arguments, parse_constant=reject_constant)
            except ValueError as e:
                raise ValueError(f"工具 {call.function.name} 的参数无效：{e}") from e

            if not isinstance(tool_input, dict):
                raise ValueError(f"工具调用参数不是字典类型: {tool_input}")

            tool_calls.append(ToolCall(
                tool_call_id = call.id,
                tool_name = call.function.name,
                tool_input = tool_input,
            ))

        # 3. 检查结束原因
        finish_reason = choice.finish_reason
        if finish_reason not in ("stop", "length", "tool_calls"):
            raise ValueError(f"不支持的结束原因: {finish_reason}")

        # 4. 构造 LLMResponse 实例
        return LLMResponse(
            content=message.content,
            tool_calls=tool_calls,
            finish_reason=finish_reason,
            usage=(raw_response.usage.model_dump() if raw_response.usage is not None else None),
        )


def reject_constant(value: str):
    raise ValueError(f"工具参数包含非标准 JSON 数值：{value}")


if __name__ == "__main__" :
    client = AsyncOpenAI(
        api_key=os.environ.get("deepseek_api_key"),
        base_url=os.environ.get("deepseek_api_base_url", "https://api.deepseek.com"),
    )

    provider = DeepSeekProvider(client)
    