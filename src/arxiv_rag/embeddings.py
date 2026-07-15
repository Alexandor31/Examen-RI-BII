from __future__ import annotations

from functools import cached_property
from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingService:
    """Lazy sentence-transformer wrapper with normalized embeddings."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    @cached_property
    def model(self) -> SentenceTransformer:
        return SentenceTransformer(self.model_name)

    @property
    def dimension(self) -> int:
        return int(self.model.get_sentence_embedding_dimension())

    def encode(
        self,
        texts: Sequence[str],
        *,
        batch_size: int = 128,
        show_progress: bool = False,
    ) -> np.ndarray:
        if not texts:
            return np.empty((0, self.dimension), dtype=np.float32)
        embeddings = self.model.encode(
            list(texts),
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        return np.asarray(embeddings, dtype=np.float32)

    def encode_query(self, query: str) -> np.ndarray:
        return self.encode([query], batch_size=1)[0]
