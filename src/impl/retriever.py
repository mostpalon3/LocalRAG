from interface.base_datastore import BaseDatastore
from interface.base_retriever import BaseRetriever

from config import TOP_K
from util.invoke_ai import invoke_ai


class Retriever(BaseRetriever):
    def __init__(self, datastore: BaseDatastore):
        self.datastore = datastore

    def rewrite_query(self, query: str) -> str:
        rewritten_query = invoke_ai(
            system_message="You rewrite user questions for document retrieval.",
            user_message=(
                "Rewrite the following query to be more specific and search-friendly "
                "for a document retrieval system. Return only the rewritten query.\n\n"
                f"Original query: {query}"
            ),
        )

        normalized_query = rewritten_query.strip()
        if not normalized_query:
            return query

        first_line = normalized_query.splitlines()[0].strip().strip('"').strip("'")
        return first_line or query

    def search(self, query: str, top_k: int = TOP_K) -> list:
        rewritten_query = self.rewrite_query(query)
        # Keep the pipeline fully local by using the datastore ranking directly.
        return self.datastore.search(rewritten_query, top_k=top_k)
