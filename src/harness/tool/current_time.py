from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from harness.tool.models import ToolDefinition

class CurrentTimeTool:
    def __init__(self) -> None:
        self.definition = ToolDefinition(
            name="get_current_time",
            description=(
                "获取指定时区的当前日期和时间。"
                "需要知道现在的时间时调用此工具。"
                "未指定时区时，默认使用 Asia/Shanghai。"
            ),
            input_schema={
                "type":"object",
                "properties":{
                    "timezone": {
                        "type": "string",
                        "description": (
                            "时区名称，例如 'Asia/Shanghai'、'America/New_York' 等。"
                            "如果未指定，将使用默认时区 'Asia/Shanghai'。"
                        ),
                        "default": "Asia/Shanghai"
                    },
                },
                "required": ["timezone"],
                "additionalProperties": False
            }
        )

    async def execute(self, arguments: dict[str, Any]) -> str:
        timezone_name = arguments.get("timezone", "Asia/Shanghai")
        now = datetime.now(ZoneInfo(timezone_name))
        return now.isoformat(timespec="seconds")
