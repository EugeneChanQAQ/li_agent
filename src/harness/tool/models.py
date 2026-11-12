from typing import Any

from pydantic import BaseModel, Field

class ToolDefinition(BaseModel):
    name: str = Field(min_length=1, description="Name of the tool")
    description: str  = Field(min_length=1, description="Description of the tool")
    input_schema: dict[str, Any] = Field(description="Schema for the tool's input parameters")
    # output_schema: dict[str, Any] | None = Field(default=None, description="Schema for the tool's output parameters")