"""
utils/pdf_loader.py
---------------------
Loads story text from uploaded TXT or PDF files, returning plain text.
"""

import logging
from pathlib import Path
from typing import Union

from pypdf import PdfReader

logger = logging.getLogger(__name__)


def load_txt(file_path: Union[str, Path]) -> str:
    """Load plain text from a .txt file."""
    path = Path(file_path)
    logger.info(f"Loading TXT file: {path}")
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        text = f.read()
    return text


def load_pdf(file_path: Union[str, Path]) -> str:
    """Extract plain text from a .pdf file, page by page."""
    path = Path(file_path)
    logger.info(f"Loading PDF file: {path}")
    reader = PdfReader(str(path))
    pages_text = []
    for i, page in enumerate(reader.pages):
        try:
            pages_text.append(page.extract_text() or "")
        except Exception as e:
            logger.warning(f"Failed to extract text from page {i}: {e}")
    return "\n\n".join(pages_text)


def load_story(file_path: Union[str, Path]) -> str:
    """
    Load a story file, dispatching to the right loader based on extension.

    Args:
        file_path: path to a .txt or .pdf file.

    Returns:
        Plain text content of the story.

    Raises:
        ValueError: if the file extension is not supported.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".txt":
        return load_txt(path)
    elif suffix == ".pdf":
        return load_pdf(path)
    else:
        raise ValueError(f"Unsupported file type: {suffix}. Use .txt or .pdf.")
