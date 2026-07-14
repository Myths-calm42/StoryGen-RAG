"""
utils/embedding.py
---------------------
Generates embeddings for story chunks using Sentence Transformers.
"""

import logging
from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from .chunking import Chunk

logger = logging.getLogger(__name__)

DEFAULT_MODEL_NAME = "BAAI/bge-small-en-v1.5"


class EmbeddingModel:
    """Wraps a Sentence Transformers model for embedding story chunks and queries."""

    def __init__(self, model_name: str = DEFAULT_MODEL_NAME):
        logger.info(f"Loading embedding model: {model_name}")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def embed_chunks(self, chunks: List[Chunk]) -> np.ndarray:
        """Embed a list of Chunk objects, returning a (N, dim) float32 array."""
        texts = [c.text for c in chunks]
        logger.info(f"Embedding {len(texts)} chunks")
        embeddings = self.model.encode(
            texts, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True
        )
        return embeddings.astype("float32")

    def embed_query(self, query: str) -> np.ndarray:
        """Embed a single query string, returning a (dim,) float32 array."""
        embedding = self.model.encode(
            [query], convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=True
        )
        return embedding.astype("float32")[0]

    @property
    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()
