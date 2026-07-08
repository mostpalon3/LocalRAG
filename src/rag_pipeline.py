from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import time
from typing import Dict, List, Optional
from interface import (
    DataItem,
    BaseDatastore,
    BaseIndexer,
    BaseRetriever,
    BaseResponseGenerator,
    BaseEvaluator,
    EvaluationResult,
)


@dataclass
class RAGPipeline:
    """Main RAG pipeline that orchestrates all components."""

    datastore: BaseDatastore
    indexer: BaseIndexer
    retriever: BaseRetriever
    response_generator: BaseResponseGenerator
    evaluator: Optional[BaseEvaluator] = None

    def reset(self) -> None:
        """Reset the datastore."""
        self.datastore.reset()

    def get_document_count(self) -> int:
        return self.datastore.get_document_count()

    def get_chunk_count(self) -> int:
        return self.datastore.get_chunk_count()

    def add_documents(self, documents: List[str]) -> None:
        """Index a list of documents."""
        indexing_started = time.perf_counter()
        items = self.indexer.index(documents)
        indexing_elapsed = time.perf_counter() - indexing_started
        print(f"[TIMING] Indexing: {indexing_elapsed:.3f}s")

        persistence_started = time.perf_counter()
        self.datastore.add_items(items)
        persistence_elapsed = time.perf_counter() - persistence_started
        print(f"[TIMING] Persistence: {persistence_elapsed:.3f}s")
        print(f"✅ Added {len(items)} items to the datastore.")

    def process_query(self, query: str) -> str:
        return self.process_query_with_details(query).response

    def process_query_with_details(self, query: str) -> "QueryResult":
        retrieval_started = time.perf_counter()
        search_results = self.retriever.search(query)
        retrieval_elapsed = time.perf_counter() - retrieval_started
        print(f"[TIMING] Retrieval: {retrieval_elapsed:.3f}s")
        print(f"✅ Found {len(search_results)} results for query: {query}\n")

        for i, result in enumerate(search_results):
            print(f"🔍 Result {i+1}: {result.source}\n")

        generation_started = time.perf_counter()
        response = self.response_generator.generate_response(query, search_results)
        generation_elapsed = time.perf_counter() - generation_started
        print(f"[TIMING] Generation: {generation_elapsed:.3f}s")

        return QueryResult(
            query=query,
            retrieved_chunks=search_results,
            response=response,
            timings={
                "retrieval": retrieval_elapsed,
                "generation": generation_elapsed,
                "total": retrieval_elapsed + generation_elapsed,
            },
        )

    def evaluate(
        self, sample_questions: List[Dict[str, str]]
    ) -> List[EvaluationResult]:
        # Evaluate a list of question/answer pairs.
        questions = [item["question"] for item in sample_questions]
        expected_answers = [item["answer"] for item in sample_questions]

        with ThreadPoolExecutor(max_workers=10) as executor:
            results: List[EvaluationResult] = list(
                executor.map(
                    self._evaluate_single_question,
                    questions,
                    expected_answers,
                )
            )

        for i, result in enumerate(results):
            result_emoji = "✅" if result.is_correct else "❌"
            print(f"{result_emoji} Q {i+1}: {result.question}: \n")
            print(f"Response: {result.response}\n")
            print(f"Expected Answer: {result.expected_answer}\n")
            print(f"Reasoning: {result.reasoning}\n")
            print("--------------------------------")

        number_correct = sum(result.is_correct for result in results)
        print(f"✨ Total Score: {number_correct}/{len(results)}")
        return results

    def _evaluate_single_question(
        self, question: str, expected_answer: str
    ) -> EvaluationResult:
        # Evaluate a single question/answer pair.
        response = self.process_query(question)
        return self.evaluator.evaluate(question, response, expected_answer)


@dataclass
class QueryResult:
    query: str
    retrieved_chunks: List[DataItem]
    response: str
    timings: Dict[str, float]
