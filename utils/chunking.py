"""
utils/chunking.py
--------------------
Splits story text into overlapping chunks, respecting paragraph
boundaries whenever possible, for embedding + retrieval.
"""

import logging
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class Chunk:
    """A single chunk of story text with metadata."""
    text: str
    chunk_id: int
    start_char: int
    end_char: int


def split_into_paragraphs(text: str) -> List[str]:
    """Split text into paragraphs on blank lines, dropping empty ones."""
    raw_paragraphs = text.split("\n\n")
    return [p.strip() for p in raw_paragraphs if p.strip()]


def chunk_story(
    text: str,
    chunk_size: int = 800,
    overlap: int = 150,
) -> List[Chunk]:
    """
    Split story text into chunks of approximately `chunk_size` characters,
    with `overlap` characters of overlap between consecutive chunks.
    Tries to break on paragraph boundaries rather than mid-sentence.

    Args:
        text: full story text.
        chunk_size: target chunk size in characters.
        overlap: number of overlapping characters between consecutive chunks.

    Returns:
        List of Chunk objects.
    """
    if chunk_size <= overlap:
        raise ValueError("chunk_size must be greater than overlap")

    paragraphs = split_into_paragraphs(text)
    if not paragraphs:
        logger.warning("No paragraphs found; falling back to raw character chunking")
        return _chunk_raw(text, chunk_size, overlap)

    chunks: List[Chunk] = []
    current_text = ""
    current_start = 0
    cursor = 0  # tracks position in the original text (approximate)
    chunk_id = 0

    for para in paragraphs:
        candidate = (current_text + "\n\n" + para).strip() if current_text else para

        if len(candidate) > chunk_size and current_text:
            # Flush current chunk
            chunks.append(Chunk(
                text=current_text,
                chunk_id=chunk_id,
                start_char=current_start,
                end_char=current_start + len(current_text),
            ))
            chunk_id += 1

            # Start next chunk with overlap: take the tail of the previous
            # chunk so context isn't lost at the boundary.
            overlap_text = current_text[-overlap:] if len(current_text) > overlap else current_text
            current_start = current_start + len(current_text) - len(overlap_text)
            current_text = (overlap_text + "\n\n" + para).strip()
        else:
            current_text = candidate

    if current_text:
        chunks.append(Chunk(
            text=current_text,
            chunk_id=chunk_id,
            start_char=current_start,
            end_char=current_start + len(current_text),
        ))

    logger.info(f"Split story into {len(chunks)} chunks (chunk_size={chunk_size}, overlap={overlap})")
    return chunks


def _chunk_raw(text: str, chunk_size: int, overlap: int) -> List[Chunk]:
    """Fallback: naive character-based chunking when no paragraphs are found."""
    chunks = []
    step = chunk_size - overlap
    for i, start in enumerate(range(0, len(text), step)):
        chunk_text = text[start:start + chunk_size]
        if not chunk_text.strip():
            continue
        chunks.append(Chunk(text=chunk_text, chunk_id=i, start_char=start, end_char=start + len(chunk_text)))
    return chunks
