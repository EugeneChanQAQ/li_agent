from typing import Self

from pydantic import BaseModel, model_validator, AwareDatetime, Field, field_validator
# from datetime import datetime

from harness.provider.models import Message, ToolMessage

class SessionRecord(BaseModel):
    session_id: str = Field(min_length=1)
    created_at: AwareDatetime # 用 AwareDatetime 代替 datetime，确保时间带时区信息
    updated_at: AwareDatetime
    messages: list[Message | ToolMessage] = Field(default_factory=list)

    @field_validator("session_id")
    @classmethod
    def validate_session_id(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("session_id 不能为空。")
        return value

    @model_validator(mode="after")
    def validate_timestamps(self) -> Self:
        if self.created_at > self.updated_at:
            raise ValueError("updated_at 不能早于 created_at。")
        return self