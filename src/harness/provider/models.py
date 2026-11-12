from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator
from harness.tool.models import ToolDefinition


class ToolMessage(BaseModel):
    role: Literal["tool"] = "tool"
    tool_name: str = Field(min_length=1, description="Name of the tool")
    tool_call_id: str = Field(min_length=1, description="Unique identifier for the tool call")
    content: str | None = None
    metadata: dict[str, Any] | None = Field(default=None, description="Additional metadata for the tool message")


class ToolCall(BaseModel):
    tool_call_id: str = Field(min_length=1,description="Unique identifier for the tool call")
    tool_name: str = Field(min_length=1, description="Name of the tool")
    tool_input: dict[str, Any] = Field(default_factory=dict, description="Input parameters for the tool call")

class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str | None = None
    tool_calls: list[ToolCall] | None = Field(default=None, description="Tool calls associated with this message")
    metadata: dict[str, Any] | None = Field(default=None, description="Additional metadata for the message")

    # Message中只有assistant可以携带非空的tool_call, 并且assistant不能同时没有tool_call和content
    # Solution：Pydantic提供装饰器 @model_validator 在字段校验后来检查跨字段规则
    # 关于不同供应商的响应数据格式问题 --> 可以单独针对每个供应商响应的格式分别创建脚本进行转换
    @model_validator(mode="after")
    def validate_tool_calls(self):
        has_content = self.content is not None and len(self.content.strip()) > 0
        has_tool_calls = self.tool_calls is not None and len(self.tool_calls) > 0
        if self.role != "assistant" and has_tool_calls:
            raise ValueError("Only assistant messages can have non-empty tool_calls.")
        if self.role == "assistant" and not has_tool_calls and not has_content:
            raise ValueError("Assistant messages must have either content or non-empty tool_calls.")

        return self
    
        
class LLMRequest(BaseModel):
    model: str = Field(min_length=1, description="Name of the LLM model")
    messages: list[Message | ToolMessage] = Field(min_length=1, description="List of messages in this turn")
    tools: list[ToolDefinition] = Field(default_factory=list, description="List of available tools")

class LLMResponse(BaseModel):
    content: str | None = Field(default=None, description="Content of the LLM response")
    tool_calls: list[ToolCall] = Field(default_factory=list, description="Tool calls associated with this response")
    finish_reason: Literal["stop", "length", "tool_calls"] | None
    usage: dict[str, Any] | None = Field(default=None, description="Usage information for the LLM response")
    # content和tool_calls不能同时为空
    @model_validator(mode='after')
    def validate_content_or_tool_calls(self):
        has_content = self.content is not None and len(self.content.strip()) > 0
        has_tool_calls = self.tool_calls is not None and len(self.tool_calls) > 0
        if not has_content and not has_tool_calls:
            if self.finish_reason != "length":
                raise ValueError("LLMResponse must have either content or non-empty tool_calls.")
        if has_content and not has_tool_calls and self.finish_reason == "tool_calls":
            raise ValueError("If LLMResponse has content but no tool_calls, finish_reason must be 'stop'.")
        return self