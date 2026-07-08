import json
import os
from pathlib import Path
from typing import List
from rag_pipeline import RAGPipeline
from create_parser import create_parser

from impl import Datastore, Indexer, Retriever, ResponseGenerator, Evaluator
from util.file_discovery import collect_files


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_SOURCE_PATH = str(ROOT_DIR)
DEFAULT_EVAL_PATH = str(ROOT_DIR / "sample_data" / "eval" / "sample_questions.json")


def create_pipeline() -> RAGPipeline:
    """Create and return a new RAG Pipeline instance with all components."""
    datastore = Datastore()
    indexer = Indexer()
    retriever = Retriever(datastore=datastore)
    response_generator = ResponseGenerator()
    evaluator = Evaluator()
    return RAGPipeline(datastore, indexer, retriever, response_generator, evaluator)


def main():
    parser = create_parser()  # Create the CLI parser
    args = parser.parse_args()
    pipeline = create_pipeline()

    # Process source paths and eval path
    source_path = args.path if args.path else DEFAULT_SOURCE_PATH
    eval_path = args.eval_file if args.eval_file else DEFAULT_EVAL_PATH
    document_paths = get_files_in_directory(source_path)

    # Execute commands
    if args.command in ["reset", "run"]:
        print("🗑️  Resetting the database...")
        pipeline.reset()

    if args.command in ["add", "run"]:
        print(f"🔍 Adding documents: {', '.join(document_paths)}")
        pipeline.add_documents(document_paths)

    if args.command in ["evaluate", "run"]:
        print(f"📊 Evaluating using questions from: {eval_path}")
        with open(eval_path, "r") as file:
            sample_questions = json.load(file)
        pipeline.evaluate(sample_questions)

    if args.command == "query":
        print(f"✨ Response: {pipeline.process_query(args.prompt)}")


def get_files_in_directory(source_path: str) -> List[str]:
    return collect_files(source_path)


if __name__ == "__main__":
    main()
