from __future__ import annotations

import math
import time
from functools import cached_property

from sentence_transformers import CrossEncoder

from .config import Settings
from .embeddings import EmbeddingService
from .models import Evidence, RetrievalOutcome
from .vector_store import VectorStore


class Retriever:
    """Dense retrieval followed by cross-encoder re-ranking."""

    def __init__(
        self,
        settings: Settings,
        embedder: EmbeddingService | None = None,
        store: VectorStore | None = None,
    ):
        self.settings = settings
        self.embedder = embedder or EmbeddingService(settings.embedding_model)
        self.store = store or VectorStore(
            settings.chroma_dir,
            settings.collection_name,
            settings.embedding_model,
        )

    @cached_property
    def reranker(self) -> CrossEncoder:
        return CrossEncoder(self.settings.reranker_model)

    @staticmethod
    def _sigmoid(value: float) -> float:
        if value >= 0:
            return 1.0 / (1.0 + math.exp(-value))
        exp_value = math.exp(value)
        return exp_value / (1.0 + exp_value)

    def retrieve(self, query: str) -> RetrievalOutcome:
        query = query.strip()
        if not query:
            raise ValueError("The query cannot be empty")
        started = time.perf_counter()
        query_vector = self.embedder.encode_query(query)
        candidates = self.store.query(query_vector, self.settings.retrieval_candidates)
        warning: str | None = None

        if self.settings.enable_reranker and candidates:
            pairs = [
                (
                    query,
                    f"Title: {item['metadata'].get('title', '')}\n"
                    f"Abstract fragment: {item['text']}",
                )
                for item in candidates
            ]
            try:
                scores = self.reranker.predict(pairs, show_progress_bar=False)
                for item, score in zip(candidates, scores):
                    item["rerank_raw"] = float(score)
                    item["rerank_score"] = self._sigmoid(float(score))
                candidates.sort(key=lambda item: item["rerank_raw"], reverse=True)
            except Exception as exc:
                warning = f"Re-ranking was unavailable; semantic order was used ({exc})."

        selected: list[dict] = []
        per_document: dict[str, int] = {}
        for item in candidates:
            doc_id = str(item["metadata"].get("doc_id", ""))
            count = per_document.get(doc_id, 0)
            if count >= self.settings.max_chunks_per_document:
                continue
            selected.append(item)
            per_document[doc_id] = count + 1
            if len(selected) >= self.settings.top_k:
                break

        evidence = tuple(
            Evidence(
                evidence_id=f"E{rank}",
                rank=rank,
                chunk_id=str(item["chunk_id"]),
                doc_id=str(item["metadata"].get("doc_id", "")),
                title=str(item["metadata"].get("title", "Untitled")),
                categories=tuple(
                    part.strip()
                    for part in str(item["metadata"].get("categories", "")).split(",")
                    if part.strip()
                ),
                text=str(item["text"]),
                semantic_score=float(item["semantic_score"]),
                rerank_score=item.get("rerank_score"),
            )
            for rank, item in enumerate(selected, start=1)
        )

        max_semantic = max(
            (float(item["semantic_score"]) for item in candidates), default=-1.0
        )
        rerank_values = [
            float(item["rerank_score"])
            for item in candidates
            if item.get("rerank_score") is not None
        ]
        insufficient = not evidence or max_semantic < self.settings.min_semantic_score
        if rerank_values:
            insufficient = insufficient or max(rerank_values) < self.settings.min_rerank_score

        elapsed_ms = (time.perf_counter() - started) * 1000
        return RetrievalOutcome(
            query=query,
            evidence=evidence,
            insufficient=insufficient,
            retrieval_ms=elapsed_ms,
            warning=warning,
        )
