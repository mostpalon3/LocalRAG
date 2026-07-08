from abc import ABC, abstractmethod
from typing import List

from .base_datastore import DataItem


class BaseResponseGenerator(ABC):

    @abstractmethod
    def generate_response(self, query: str, context: List[DataItem]) -> str:
        pass
