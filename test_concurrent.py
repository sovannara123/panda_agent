import asyncio
import time

from async_agent import AsyncAgent
from storage import MemoryStorage


async def test_concurrent_requests():
    """Test handling multiple requests concurrently with separate agents."""
    
    questions = [
        "What is the price of laptop?",
        "Check order A100",
        "Tell me about Python"
    ]

    async def ask(question):
        agent = AsyncAgent(storage=MemoryStorage(f'/tmp/test_{id(question)}.json', f'/tmp/meta_{id(question)}.json'))
        return await agent.respond_async(question)

    start = time.time()

    # Run all requests concurrently
    tasks = [ask(q) for q in questions]
    results = await asyncio.gather(*tasks)

    elapsed = time.time() - start

    print(f"\nProcessed {len(questions)} requests in {elapsed:.2f} seconds")

    for i, (question, response) in enumerate(zip(questions, results), 1):
        print(f"\nQ{i}: {question}")
        print(f"A{i}: {response[:100]}...")

    return results


if __name__ == "__main__":
    asyncio.run(test_concurrent_requests())