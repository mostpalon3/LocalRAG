from abc import ABC, abstractmethod
from typing import List

from .base_datastore import DataItem


class BaseRetriever(ABC):

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> List[DataItem]:
        pass
