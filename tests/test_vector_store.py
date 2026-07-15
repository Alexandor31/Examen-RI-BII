from __future__ import annotations

import numpy as np

from arxiv_rag.models import Chunk
from arxiv_rag.vector_store import VectorStore


def make_chunk(index: int) -> Chunk:
    return Chunk(
        chunk_id=f"doc-{index}:c000",
        doc_id=f"doc-{index}",
        title=f"Paper {index}",
        categories=("cs.AI",),
        text=f"Abstract {index}",
        chunk_index=0,
        total_chunks=1,
    )


def test_vector_store_persists_and_queries_cosine_vectors(tmp_path) -> None:
    store = VectorStore(tmp_path / "chroma", "test_collection", "test-model")
    chunks = [make_chunk(1), make_chunk(2)]
    embeddings = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    store.upsert(chunks, embeddings)

    reopened = VectorStore(tmp_path / "chroma", "test_collection", "test-model")
    results = reopened.query(np.asarray([0.95, 0.05], dtype=np.float32), n_results=2)

    assert reopened.count == 2
    assert reopened.existing_ids() == {"doc-1:c000", "doc-2:c000"}
    assert results[0]["chunk_id"] == "doc-1:c000"
    assert results[0]["semantic_score"] > results[1]["semantic_score"]
