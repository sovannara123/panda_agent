from config import AGENT_NAME, MESSAGE_LIMIT_FREE
from llm import generate_response
from storage import load_memory, save_memory
from tools import get_intent, get_response


class Agent:
    def __init__(self):
        self.memory = load_memory()

    def check_access(self):
        if self.memory["user_plan"] == "premium":
            return True
        return self.memory["message_used"] < MESSAGE_LIMIT_FREE

    def respond(self, user_msg):
        intent = get_intent(user_msg)

        self.memory["conversation"].append({"role": "user", "content": user_msg})

        response = generate_response(self.memory["conversation"])
        if response is None:
            response = get_response(intent)

        self.memory["conversation"].append({"role": "assistant", "content": response})
        self.memory["message_used"] += 1

        return response

    def run(self):
        print(f"Hello, I am {AGENT_NAME}. How can I help you?")
        while True:
            user_msg = input("You: ").strip()
            if user_msg.lower() in ("quit", "exit"):
                print("Goodbye!")
                break
            if not self.check_access():
                print("System: Message limit reached.")
                break

            response = self.respond(user_msg)
            print(f"{AGENT_NAME}: {response}")

        save_memory(self.memory)