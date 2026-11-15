import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from harness.provider.deepseek_provider import DeepSeekProvider
from harness.runtime.loop import AgentLoop
from harness.runtime.runtime import AgentRuntime
from harness.runtime.store import MemoryStore
from harness.session.chat_session import ChatSession
from harness.tool.current_time import CurrentTimeTool
from harness.tool.executor import ToolExecutor


async def main() -> None:
    # 读取配置。
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

    async with AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
    ) as client:
        # 组装执行链，整个会话期间复用。
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

        # 会话只创建一次，负责跨 Run 保存历史。
        session = ChatSession(
            runtime=runtime,
            model=model,
            tools=[time_tool.definition],
        )

        print("会话已启动。")
        print("输入 /clear 清空历史，输入 /exit 退出。")

        while True:
            try:
                text = input("\n请输入任务：").strip()
            except EOFError:
                print("\n会话已结束。")
                break

            if text == "/exit":
                print("会话已结束。")
                break

            if text == "/clear":
                session.clear()
                print("会话历史已清空。")
                continue

            if not text:
                print("消息不能为空。")
                continue

            print("\n执行中……")

            try:
                record = await session.send_message(text)
            except Exception as exc:
                # 处理 Runtime 向外抛出的异常，例如存储失败。
                print(f"\n执行异常：{type(exc).__name__}: {exc}")
                print("本轮未加入会话历史。")
                continue

            print(f"\n执行 ID：{record.run_id}")
            print(f"执行状态：{record.status}")

            if record.result is not None:
                print(f"停止原因：{record.result.stop_reason}")

                if record.result.final_text is not None:
                    print("\n回答：")
                    print(record.result.final_text)
                else:
                    print("\n本次执行没有最终回答。")

            if record.error is not None:
                print(f"\n执行错误：{record.error}")

            if record.status != "completed":
                print("本轮未加入会话历史。")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n已中断运行。")