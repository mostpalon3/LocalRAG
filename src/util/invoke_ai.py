import os
import ollama


def invoke_ai(system_message: str, user_message: str) -> str:
    """
    Generic function to invoke an AI model given a system and user message.
    Replace this if you want to use a different AI model.
    """

    client = ollama.Client(host=os.getenv("OLLAMA_HOST", "http://localhost:11434"))
    response = client.chat(
        model=os.getenv("OLLAMA_CHAT_MODEL", "qwen3:8b"),
        messages=[
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ],
    )
    return response["message"]["content"]
