import os
from pathlib import Path
from typing import List
from interface.base_datastore import DataItem
from interface.base_indexer import BaseIndexer
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker, DocChunk

from config import MAX_TOKENS
from util.extract_content import (
    extract_image_text,
    extract_pdf_text,
    extract_raw_text,
    is_image_file,
    is_text_file,
)


class Indexer(BaseIndexer):
    def __init__(self):
        self.converter = DocumentConverter()
        self.chunker = HybridChunker(max_tokens=MAX_TOKENS)
        # Disable tokenizers parallelism to avoid OOM errors.
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def index(self, document_paths: List[str]) -> List[DataItem]:
        items = []
        for document_path in document_paths:
            items.extend(self._index_single_file(document_path))
        return items

    def _index_single_file(self, document_path: str) -> List[DataItem]:
        if is_image_file(document_path):
            ocr_text = extract_image_text(document_path)
            return self._items_from_text(ocr_text, document_path)

        if Path(document_path).suffix.lower() == ".pdf":
            docling_items = self._index_with_docling(document_path)
            if docling_items:
                return docling_items

            ocr_text = extract_pdf_text(document_path)
            return self._items_from_text(ocr_text, document_path)

        if is_text_file(document_path):
            raw_text = extract_raw_text(document_path)
            return self._items_from_text(raw_text, document_path)

        docling_items = self._index_with_docling(document_path)
        if docling_items:
            return docling_items

        raw_text = extract_raw_text(document_path)
        return self._items_from_text(raw_text, document_path)

    def _index_with_docling(self, document_path: str) -> List[DataItem]:
        try:
            document = self.converter.convert(document_path).document
            chunks: List[DocChunk] = self.chunker.chunk(document)
            if not chunks:
                return []

            return self._items_from_chunks(chunks, document_path)
        except Exception:
            return []

    def _items_from_chunks(
        self, chunks: List[DocChunk], document_path: str
    ) -> List[DataItem]:
        items = []
        source_prefix = os.path.abspath(document_path)
        for i, chunk in enumerate(chunks):
            headings = ", ".join(chunk.meta.headings) if chunk.meta.headings else ""
            content_headings = "## " + headings if headings else "##"
            content_text = f"{content_headings}\n{chunk.text}"
            source = f"{source_prefix}:{i}"
            item = DataItem(content=content_text, source=source)
            items.append(item)

        return items

    def _items_from_text(self, raw_text: str, document_path: str) -> List[DataItem]:
        text = raw_text.strip()
        if not text:
            return []

        max_characters = max(1, MAX_TOKENS * 4)
        text_chunks = [
            text[start : start + max_characters]
            for start in range(0, len(text), max_characters)
        ]
        source_prefix = os.path.abspath(document_path)
        file_name = Path(document_path).name

        return [
            DataItem(
                content=f"## {file_name}\n{chunk_text}",
                source=f"{source_prefix}:{index}",
            )
            for index, chunk_text in enumerate(text_chunks)
        ]

