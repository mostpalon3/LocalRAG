from typing import List
from interface.base_datastore import DataItem
from interface.base_response_generator import BaseResponseGenerator
from util.invoke_ai import invoke_ai


SYSTEM_PROMPT = """
Use the provided context to provide a concise answer to the user's question.
If you cannot find the answer in the context, say so. Do not make up information.
"""


class ResponseGenerator(BaseResponseGenerator):
    def build_user_message(self, query: str, context: List[DataItem]) -> str:
        context_text = "\n".join(item.content for item in context)
        return (
            f"<context>\n{context_text}\n</context>\n"
            f"<question>\n{query}\n</question>"
        )

    def generate_response(self, query: str, context: List[DataItem]) -> str:
        """Generate a response using the configured chat model."""
        user_message = self.build_user_message(query, context)

        return invoke_ai(system_message=SYSTEM_PROMPT, user_message=user_message)

    def stream_response(self, query: str, context: List[DataItem]):
        user_message = self.build_user_message(query, context)
        return invoke_ai(system_message=SYSTEM_PROMPT, user_message=user_message, stream=True)
