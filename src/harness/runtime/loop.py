from harness.runtime.models import LoopResult
from harness.provider.models import LLMRequest, Message, ToolMessage

class AgentLoop:
    def __init__(self, provider, tool_executor, max_steps: int = 10):
        if max_steps <= 0:
            raise ValueError("max_steps 必须大于 0")

        self.provider = provider
        self.tool_executor = tool_executor
        self.max_steps = max_steps

    async def run(self, request: LLMRequest) -> LoopResult:

        working_request = request.model_copy(deep=True)
        messages: list[Message | ToolMessage] = working_request.messages
        allowed_tool_names = frozenset(
            tool.name for tool in working_request.tools
        )

        for _ in range(self.max_steps):
            # 用当前消息构建本轮请求
            # 调用provider
            # 检查响应并追加 assistant 消息
            # 根据结束原因返回，或者执行工具
            current_request = LLMRequest(
                model=working_request.model,
                messages=messages,
                tools=working_request.tools,
            )

            response = await self.provider.generate(current_request)

            # finish_reason 为 length 时，v1.0 先简单处理 LoopResult的信息
            # 这里注意要放在messages.append前面，不然可能会导致截断的 message 流入占用上下文且影响模型（具体看 Runtime 如何复用）
            # 后续要处理，因为 Runtime 后续可能会直接把 LoopResult.messages 全部作为新请求的历史

            # 输出截断时直接结束，不将本轮响应加入消息历史。
            # 避免后续复用 messages 时，将部分输出或未执行的工具调用带入模型上下文。
            if response.finish_reason == "length":
                return LoopResult(
                    messages=messages,
                    final_text=None,
                    stop_reason="output_truncated",
                )

            messages.append(
                Message(
                    role="assistant",
                    content=response.content,
                    tool_calls=response.tool_calls
                )
            )

            # 维护 LoopResult 中的 Message 部分

            # finish_reason 为 stop 时，本次 loop 结束，返回 Loop 结果
            if response.finish_reason == "stop":
                return LoopResult(
                    messages=messages,
                    final_text=response.content,
                    stop_reason="completed",
                )

            # finish_reason 为 tool_calls 时，进行工具调用
            # 1. 追加 assistant 消息，记录模型提出的工具调用
            # 2. 遍历 tool_calls，逐个交给工具执行器（v1.0 还未考虑工具能否并行、是否互相依赖的问题）
            # 3. Tool 返回结果追加到 messages 中
            # 4. 进入下一轮模型调用，让模型看到工具结果
            # Tips: 如果最后一轮是调用工具，那应返回max_steps，并不会再调用模型
            elif response.finish_reason == "tool_calls":
                tool_list = response.tool_calls

                for tool_call in tool_list:
                    tool_message = await self.tool_executor.execute(tool_call, allowed_tool_names=allowed_tool_names)
                    messages.append(tool_message)

        # 超过最大循环次数，返回 LoopResult
        return LoopResult(
            messages=messages,
            final_text=None,
            stop_reason="max_steps",
        )
