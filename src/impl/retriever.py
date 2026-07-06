from interface.base_datastore import BaseDatastore
from interface.base_retriever import BaseRetriever


class Retriever(BaseRetriever):
    def __init__(self, datastore: BaseDatastore):
        self.datastore = datastore

    def search(self, query: str, top_k: int = 10) -> list[str]:
        # Keep the pipeline fully local by using the datastore ranking directly.
        return self.datastore.search(query, top_k=top_k)
