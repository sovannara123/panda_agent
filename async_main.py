import asyncio
import uuid

from async_agent import AsyncAgent


async def main():
    agent = AsyncAgent()
    session_id = str(uuid.uuid4())

    print(f"{agent.name} is ready.")
    print(f"Session ID: {session_id}")
    print("Type 'quit' or 'exit' to stop.")

    while True:
        try:
            user_input = input("You: ")
        except (KeyboardInterrupt, EOFError):
            break

        if user_input.lower().strip() in {"quit", "exit"}:
            break

        if not user_input.strip():
            print("Agent: Please type a message.")
            continue

        # Stream response
        print("Agent: ", end="", flush=True)

        async for token in agent.respond_stream_async(user_input, session_id=session_id):
            print(token, end="", flush=True)    

        print()

    print("\nGoodbye!")


if __name__ == "__main__":
    asyncio.run(main())