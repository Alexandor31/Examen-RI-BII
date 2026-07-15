from __future__ import annotations

from pathlib import Path
from typing import Sequence

import chromadb
import numpy as np
from chromadb.api.models.Collection import Collection

from .models import Chunk


class VectorStore:
    """Persistent Chroma collection using externally computed embeddings."""

    def __init__(
        self,
        persist_directory: Path,
        collection_name: str,
        embedding_model: str,
        *,
        allow_model_mismatch: bool = False,
    ):
        persist_directory.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=str(persist_directory))
        self.collection_name = collection_name
        self.embedding_model = embedding_model
        self.allow_model_mismatch = allow_model_mismatch
        self.collection = self._get_or_create_collection()

    def _get_or_create_collection(self) -> Collection:
        collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={
                "hnsw:space": "cosine",
                "embedding_model": self.embedding_model,
                "corpus": "spsayakpaul/arxiv-paper-abstracts",
            },
        )
        configured_model = (collection.metadata or {}).get("embedding_model")
        if (
            configured_model
            and configured_model != self.embedding_model
            and not self.allow_model_mismatch
        ):
            raise ValueError(
                "The existing Chroma collection was built with "
                f"'{configured_model}', but EMBEDDING_MODEL is '{self.embedding_model}'. "
                "Rebuild it with `python scripts/build_index.py --reset`."
            )
        return collection

    @property
    def count(self) -> int:
        return self.collection.count()

    def existing_ids(self) -> set[str]:
        """Return stored IDs without loading embeddings or document payloads."""
        if self.count == 0:
            return set()
        result = self.collection.get(include=[])
        return {str(item) for item in result.get("ids", [])}

    def reset(self) -> None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception as exc:
            # Chroma raises different not-found types across versions.
            if "does not exist" not in str(exc).lower() and "not found" not in str(exc).lower():
                raise
        self.collection = self._get_or_create_collection()

    def upsert(self, chunks: Sequence[Chunk], embeddings: np.ndarray) -> None:
        if not chunks:
            return
        if len(chunks) != len(embeddings):
            raise ValueError("The chunk and embedding counts do not match")
        self.collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=embeddings,
            documents=[chunk.text for chunk in chunks],
            metadatas=[chunk.metadata for chunk in chunks],
        )

    def query(self, query_embedding: np.ndarray, n_results: int) -> list[dict]:
        if self.count == 0:
            return []
        result = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=min(n_results, self.count),
            include=["documents", "metadatas", "distances"],
        )
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        rows: list[dict] = []
        for chunk_id, document, metadata, distance in zip(
            ids, documents, metadatas, distances
        ):
            rows.append(
                {
                    "chunk_id": chunk_id,
                    "text": document or "",
                    "metadata": metadata or {},
                    "distance": float(distance),
                    "semantic_score": 1.0 - float(distance),
                }
            )
        return rows
