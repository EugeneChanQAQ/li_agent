import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from harness.provider.deepseek_provider import DeepSeekProvider
from harness.provider.models import LLMRequest, Message
from harness.runtime.loop import AgentLoop
from harness.runtime.runtime import AgentRuntime
from harness.runtime.store import MemoryStore
from harness.tool.executor import ToolExecutor
from harness.tool.current_time import CurrentTimeTool


async def main() -> None:
    # 从 demo.py 所在目录读取 .env
    # 默认不会覆盖已有的系统环境变量。
    env_path = Path(__file__).resolve().parent / ".env"
    load_dotenv(env_path)

    api_key = os.getenv("DEEPSEEK_API_KEY", "").strip()
    model = os.getenv("DEEPSEEK_MODEL", "").strip()
    base_url = (
        os.getenv("DEEPSEEK_BASE_URL", "").strip()
        or "https://api.deepseek.com"
    )

    if not api_key:
        raise ValueError("请在 .env 或环境变量中配置 DEEPSEEK_API_KEY。")
    if not model:
        raise ValueError("请在 .env 或环境变量中配置 DEEPSEEK_MODEL。")

    task = input("请输入任务：").strip()
    if not task:
        print("任务不能为空。")
        return

    async with AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
    ) as client:
        provider = DeepSeekProvider(client)

        time_tool = CurrentTimeTool()
        executor = ToolExecutor(tools=[time_tool])
        loop = AgentLoop(
            provider=provider,
            tool_executor=executor,
            max_steps=5,
        )
        store = MemoryStore()
        runtime = AgentRuntime(loop=loop, store=store)

        request = LLMRequest(
            model=model,
            messages=[Message(role="user", content=task)],
            tools=[time_tool.definition],
        )

        print("\n执行中……")
        record = await runtime.run(request)

        print(f"\n执行 ID：{record.run_id}")
        print(f"执行状态：{record.status}")

        if record.result is not None:
            print(f"停止原因：{record.result.stop_reason}")
            print("\n回答：")
            print(record.result.final_text or "本次执行没有最终回答。")

        if record.error is not None:
            print(f"\n执行错误：{record.error}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n已中断运行。")