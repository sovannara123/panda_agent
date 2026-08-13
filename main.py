from agent import Agent


def main():
    agent = Agent()

    print(f"{agent.name} is ready.")
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

        response = agent.respond(user_input)
        print("Agent:", response)


if __name__ == "__main__":
    main()