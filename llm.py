from openai import OpenAI

from config import API_KEY, MODEL_NAME


def generate_response(messages):
    if not API_KEY:
        return None

    client = OpenAI(api_key=API_KEY)
    completion = client.chat.completions.create(
        model=MODEL_NAME,
        messages=messages,
        temperature=0.7,
    )
    return completion.choices[0].message.content