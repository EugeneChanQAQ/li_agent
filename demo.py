import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from datetime import datetime, timezone
from uuid import uuid4

from harness.provider.deepseek_provider import DeepSeekProvider
from harness.runtime.loop import AgentLoop
from harness.runtime.runtime import AgentRuntime
from harness.runtime.store import MemoryStore, JsonRunStore
from harness.session.chat_session import ChatSession, SessionSaveError
from harness.tool.current_time import CurrentTimeTool
from harness.tool.executor import ToolExecutor
from harness.session.models import SessionRecord
from harness.session.store import JsonSessionStore


async def main() -> None:
    # 读取配置。
    project_path = Path(__file__).resolve().parent
    env_path = project_path / ".env"
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

    session_store = JsonSessionStore(project_path / "data" / "session.json")

    try:
        session_record = session_store.load()

        if session_record is None:
            now = datetime.now(timezone.utc)
            session_record = SessionRecord(
                session_id=str(uuid4()),
                created_at=now,
                updated_at=now,
            )
            # 保存初始空会话，让会话 ID 在重启后保持一致。
            session_store.save(session_record)

    except Exception as exc:
        print(f"会话初始化失败：{type(exc).__name__}: {exc}")
        return

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

        store = JsonRunStore(project_path / "data" / "runs")
        runtime = AgentRuntime(loop=loop, store=store)

        # 会话只创建一次，负责跨 Run 保存历史。
        # 20261117 更新 ChatSession，接入 JsonSessionStore，支持跨进程保存历史。
        session = ChatSession(
            runtime=runtime,
            model=model,
            tools=[time_tool.definition],
            store=session_store,
            record=session_record,
        )

        print(f"会话已启动：{session_record.session_id}")
        print(f"已加载 {len(session_record.messages)} 条历史消息。")
        print("输入 /clear 清空历史，输入 /exit 退出。")

        # print("会话已启动。")
        # print("输入 /clear 清空历史，输入 /exit 退出。")

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
                try:
                    session.clear()
                except Exception as exc:
                    print(f"清空失败：{type(exc).__name__}: {exc}")
                    print("内存中的会话历史未清空。")
                else:
                    print("会话历史已清空并保存。")
                continue

            if not text:
                print("消息不能为空。")
                continue

            print("\n执行中……")

            try:
                record = await session.send_message(text)

            except SessionSaveError as exc:
                # 执行成功，仍然展示回答，但明确说明未保存。
                record = exc.run_record
                print("\n本轮执行已完成，但未保存到会话历史。")
                print("后续输入将继续使用此前已保存的历史。")

                if exc.__cause__ is not None:
                    cause = exc.__cause__
                    print(
                        f"保存异常：{type(cause).__name__}: {cause}"
                    )

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