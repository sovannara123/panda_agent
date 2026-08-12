from agent import Agent
from config import AGENT_NAME, AGENT_ROLE, AGENT_VERSION


def main():
    print(f"Starting {AGENT_NAME} ({AGENT_ROLE}) v{AGENT_VERSION}")
    agent = Agent()
    agent.run()


if __name__ == "__main__":
    main()