"""
utils/retrieval.py
---------------------
FAISS-backed vector store for story chunks, supporting persistence and
top-K similarity search.
"""

import json
import logging
from pathlib import Path
from typing import List, Tuple, Union

import faiss
import numpy as np

from .chunking import Chunk
from .embedding import EmbeddingModel

logger = logging.getLogger(__name__)


class StoryVectorStore:
    """
    FAISS-backed store mapping story chunks to their embeddings, with
    similarity search and disk persistence.
    """

    def __init__(self, embedding_model: EmbeddingModel):
        self.embedding_model = embedding_model
        self.index: faiss.Index = None
        self.chunks: List[Chunk] = []

    def build(self, chunks: List[Chunk]) -> None:
        """Build a fresh FAISS index from a list of chunks."""
        self.chunks = chunks
        embeddings = self.embedding_model.embed_chunks(chunks)
        dim = embeddings.shape[1]

        # Inner product on normalized embeddings == cosine similarity
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)
        logger.info(f"Built FAISS index with {self.index.ntotal} vectors (dim={dim})")

    def search(self, query: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        """
        Retrieve the top_k most relevant chunks for a query.

        Returns:
            List of (Chunk, similarity_score) tuples, sorted by relevance.
        """
        if self.index is None or self.index.ntotal == 0:
            logger.warning("Vector store is empty; no results to retrieve")
            return []

        query_embedding = self.embedding_model.embed_query(query).reshape(1, -1)
        top_k = min(top_k, self.index.ntotal)
        scores, indices = self.index.search(query_embedding, top_k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append((self.chunks[idx], float(score)))
        return results

    def save(self, directory: Union[str, Path]) -> None:
        """Persist the FAISS index and chunk metadata to disk."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)

        faiss.write_index(self.index, str(directory / "index.faiss"))

        chunk_records = [
            {"text": c.text, "chunk_id": c.chunk_id, "start_char": c.start_char, "end_char": c.end_char}
            for c in self.chunks
        ]
        with open(directory / "chunks.json", "w", encoding="utf-8") as f:
            json.dump(chunk_records, f, ensure_ascii=False, indent=2)

        logger.info(f"Saved vector store to {directory}")

    def load(self, directory: Union[str, Path]) -> None:
        """Load a previously saved FAISS index and chunk metadata."""
        directory = Path(directory)
        self.index = faiss.read_index(str(directory / "index.faiss"))

        with open(directory / "chunks.json", "r", encoding="utf-8") as f:
            chunk_records = json.load(f)
        self.chunks = [
            Chunk(text=r["text"], chunk_id=r["chunk_id"], start_char=r["start_char"], end_char=r["end_char"])
            for r in chunk_records
        ]
        logger.info(f"Loaded vector store from {directory} ({len(self.chunks)} chunks)")
