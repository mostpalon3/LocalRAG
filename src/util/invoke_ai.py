import ollama

from config import LLM_MODEL, OLLAMA_URL


def invoke_ai(system_message: str, user_message: str, stream: bool = False):
    """
    Generic function to invoke an AI model given a system and user message.
    Replace this if you want to use a different AI model.
    """

    client = ollama.Client(host=OLLAMA_URL)
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": user_message},
    ]

    if stream:
        return (
            chunk.get("message", {}).get("content", "")
            for chunk in client.chat(model=LLM_MODEL, messages=messages, stream=True)
        )

    response = client.chat(model=LLM_MODEL, messages=messages)
    return response["message"]["content"]
