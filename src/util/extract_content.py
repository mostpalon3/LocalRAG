from functools import lru_cache
from pathlib import Path
import warnings

import easyocr
import filetype
import numpy as np
import pypdfium2
from PIL import Image, UnidentifiedImageError
from charset_normalizer import from_path


IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


@lru_cache(maxsize=1)
def get_ocr_reader() -> easyocr.Reader:
    return easyocr.Reader(["en"], gpu=False, verbose=False)


def is_image_file(document_path: str) -> bool:
    kind = _guess_file_kind(document_path)
    if kind is not None:
        return kind.mime.startswith("image/")

    return Path(document_path).suffix.lower() in IMAGE_EXTENSIONS


def is_text_file(document_path: str) -> bool:
    kind = _guess_file_kind(document_path)
    if kind is not None:
        return False

    return _looks_like_text(document_path)


def extract_image_text(document_path: str) -> str:
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("error", category=Image.DecompressionBombWarning)
            image = Image.open(document_path).convert("RGB")
    except (Image.DecompressionBombWarning, UnidentifiedImageError, OSError):
        return ""

    return _extract_text_from_array(np.array(image))


def extract_pdf_text(document_path: str) -> str:
    pdf = pypdfium2.PdfDocument(document_path)
    page_texts = []

    for page_number in range(len(pdf)):
        page = pdf[page_number]
        bitmap = page.render(scale=2)
        page_text = _extract_text_from_array(np.array(bitmap.to_pil()))
        if page_text.strip():
            page_texts.append(f"[Page {page_number + 1}]\n{page_text.strip()}")

    return "\n\n".join(page_texts)


def extract_raw_text(document_path: str) -> str:
    detected_text = from_path(document_path).best()
    if detected_text is not None:
        return str(detected_text)

    return Path(document_path).read_text(encoding="utf-8", errors="replace")


def _extract_text_from_array(image_array) -> str:
    reader = get_ocr_reader()
    results = reader.readtext(image_array, detail=0, paragraph=True)
    return "\n".join(line.strip() for line in results if line and line.strip())


def _looks_like_text(document_path: str) -> bool:
    try:
        with open(document_path, "rb") as file_handle:
            sample = file_handle.read(4096)
    except OSError:
        return False

    if not sample:
        return True

    if b"\x00" in sample:
        return False

    try:
        sample.decode("utf-8")
        return True
    except UnicodeDecodeError:
        printable_bytes = sum(
            byte == 9
            or byte == 10
            or byte == 13
            or 32 <= byte <= 126
            for byte in sample
        )
        return printable_bytes / len(sample) > 0.85


def _guess_file_kind(document_path: str):
    try:
        with open(document_path, "rb") as file_handle:
            sample = file_handle.read(261)
    except OSError:
        return None

    if not sample:
        return None

    return filetype.guess(sample)