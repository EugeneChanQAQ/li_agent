from typing import Literal
from datetime import datetime

from pydantic import BaseModel, Field, model_validator, field_validator

from harness.provider.models import ToolMessage, Message, LLMRequest

class LoopResult(BaseModel):
    messages: list[Message | ToolMessage] = Field(min_length=1, description="List of messages in this turn")
    final_text: str | None = Field(default=None, description="Final result of LLM after one turn")
    stop_reason: Literal["completed", "max_steps", "output_truncated"]

    @model_validator(mode="after")
    def final_text_check(self):
        final_answer: str | None = self.final_text
        final_message: Message | ToolMessage = self.messages[-1]
        # if self.stop_reason not in ["completed", "max_steps", "output_truncated"]:
        #     raise ValueError(f"不支持的结束原因：{self.stop_reason}")

        if self.stop_reason != "completed" and self.final_text is not None:
            raise ValueError("未正常完成时，final_text 必须为 None。")

        if self.stop_reason == "completed":
            if final_answer is None or not final_answer.strip():
                raise ValueError("任务完成必须有最终返回结果。")
            if final_message.content is None or not final_message.content.strip():
                raise ValueError("模型完成时返回内容不能为空")

            if final_message.role != "assistant":
                raise ValueError("标记完成的任务最后一条message的role必须是assistant")
            if final_message.tool_calls:
                raise ValueError("最后一条assistant返回值有代执行的tool")
            if final_answer != final_message.content:
                raise ValueError("Final answer 必须与最后一条 assistant 消息的内容一致。")
        return self

class RunRecord(BaseModel):
    run_id: str
    session_id: str
    request: LLMRequest
    status: Literal["running", "completed", "stopped", "failed"]
    started_at: datetime
    finished_at: datetime | None = None
    result: LoopResult | None = None
    error: str | None = None

    @field_validator("run_id", "session_id")
    @classmethod
    def validate_identifier(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("run_id 和 session_id 不能为空。")
        return value

    @model_validator(mode="after")
    def validate_state(self) -> "RunRecord": # 加引号是因为此时类还在定义过程中
        # 1. 时间需要带时区，避免本地时间和 UTC 混用。
        if self.started_at.utcoffset() is None:
            raise ValueError("started_at 必须携带时区。")

        if self.finished_at is not None:
            if self.finished_at.utcoffset() is None:
                raise ValueError("finished_at 必须携带时区。")
                # finished_at 时间必须迟于 started_at
            if self.finished_at < self.started_at:
                raise ValueError("finished_at 不能早于 started_at。")

        # 2. 运行中：尚未结束，也不应该有最终结果或错误
        if self.status == "running":
            if self.finished_at is not None:
                raise ValueError("running 状态下，finished_at 必须为 None。")
            if self.result is not None:
                raise ValueError("running 状态下，result 必须为 None。")
            if self.error is not None:
                raise ValueError("running 状态下，error 必须为 None。")
            return self

        # 3. 其他三种状态都代表执行已结束
        if self.finished_at is None:
            raise ValueError("执行结束时，finished_at 必须存在。")

        # 4. 失败：有错误信息，但没有正常返回 LoopResult
        if self.status == "failed":
            if self.error is None or not self.error.strip():
                raise ValueError("failed 状态下，error 必须为 非空字符串")
            if self.result is not None:
                raise ValueError("failed 状态下，result 必须为 None")

            return self

        # 5. completed 和 stopped：Loop正常返回结果
        if self.error is not None:
            raise ValueError("completed 和 stopped 状态下，error 必须为 None")

        if self.result is None:
            raise ValueError("completed 和 stopped 状态下，result 必须存在。")

        if self.status == "completed":
            if self.result.stop_reason != "completed":
                raise ValueError("completed 状态要求 result.stop_reason 为 completed。")
        elif self.status == "stopped":
            if self.result.stop_reason not in ("max_steps", "output_truncated"):
                raise ValueError("stopped 状态要求 result.stop_reason 为 "
                                    "max_steps 或 output_truncated。")

        return self
                    