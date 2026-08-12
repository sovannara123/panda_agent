def get_intent(message):
    msg = message.lower()
    if "help" in msg:
        return "help"
    elif "price" in msg or "cost" in msg:
        return "price"
    elif "learn" in msg or "teach" in msg:
        return "learn"
    elif "bye" in msg or "exit" in msg or "quit" in msg:
        return "goodbye"
    else:
        return "unknown"


def get_response(intent):
    responses = {
        "help": "I can help you with Python, AI agents, or answer questions.",
        "price": "Free tier: 10 messages. Premium: unlimited for $10/month.",
        "learn": "Great! Let's start with variables, strings, and functions.",
        "goodbye": "Goodbye! Happy coding!",
        "unknown": "I'm not sure. Try asking for help, price, or learn.",
    }
    return responses.get(intent, responses["unknown"])