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
        file_label = self._document_label(document_path)
        for i, chunk in enumerate(chunks):
            headings = ", ".join(chunk.meta.headings) if chunk.meta.headings else ""
            content_headings = "## " + headings if headings else "##"
            content_text = f"{content_headings}\n## File: {file_label}\n{chunk.text}"
            source = f"{source_prefix}:{i}"
            item = DataItem(content=content_text, source=source)
            items.append(item)

        return items

    def _items_from_text(self, raw_text: str, document_path: str) -> List[DataItem]:
        text = raw_text.strip()
        if not text:
            return []

        max_characters = max(600, MAX_TOKENS * 3)
        text_chunks = self._split_text_into_chunks(text, max_characters=max_characters)
        source_prefix = os.path.abspath(document_path)

        return [
            DataItem(
                content=(
                    f"## File: {self._document_label(document_path)}\n"
                    f"## Lines: {start_line}-{end_line}\n"
                    f"{chunk_text}"
                ),
                source=f"{source_prefix}:{start_line}-{end_line}",
            )
            for chunk_text, start_line, end_line in text_chunks
        ]

    def _document_label(self, document_path: str) -> str:
        path = Path(document_path)
        try:
            return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
        except ValueError:
            return path.name

    def _split_text_into_chunks(
        self, text: str, max_characters: int, overlap_lines: int = 8
    ) -> List[tuple[str, int, int]]:
        lines = text.splitlines()
        if not lines:
            return []

        chunks: List[tuple[str, int, int]] = []
        start_index = 0
        total_lines = len(lines)

        while start_index < total_lines:
            end_index = start_index
            current_length = 0

            while end_index < total_lines:
                line = lines[end_index]
                line_length = len(line) + 1

                if current_length and current_length + line_length > max_characters:
                    break

                current_length += line_length
                end_index += 1

                if current_length >= max_characters:
                    break

            if end_index == start_index:
                line = lines[start_index]
                for chunk_start in range(0, len(line), max_characters):
                    chunk_text = line[chunk_start : chunk_start + max_characters].strip()
                    if chunk_text:
                        chunks.append((chunk_text, start_index + 1, start_index + 1))
                start_index += 1
                continue

            chunk_text = "\n".join(lines[start_index:end_index]).strip()
            if chunk_text:
                chunks.append((chunk_text, start_index + 1, end_index))

            if end_index >= total_lines:
                break

            start_index = max(end_index - overlap_lines, start_index + 1)

        return chunks

