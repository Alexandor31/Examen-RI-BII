from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Iterable, Sequence, TypeVar

from .config import Settings
from .corpus import DATASET_URL, download_corpus, iter_chunks, load_corpus, select_corpus_file
from .embeddings import EmbeddingService
from .models import Chunk
from .vector_store import VectorStore

T = TypeVar("T")


@dataclass(frozen=True)
class IndexStats:
    source_file: str
    documents: int
    chunks: int
    collection_count: int
    embedding_model: str
    created_at: str
    reused_existing: bool = False


def _batches(items: Sequence[T], size: int) -> Iterable[Sequence[T]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _read_manifest(settings: Settings) -> dict:
    if not settings.manifest_path.exists():
        return {}
    try:
        return json.loads(settings.manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def _read_existing_stats(settings: Settings, store: VectorStore) -> IndexStats:
    manifest = _read_manifest(settings)
    return IndexStats(
        source_file=manifest.get("source_file", "unknown"),
        documents=int(manifest.get("documents", 0)),
        chunks=int(manifest.get("chunks", store.count)),
        collection_count=store.count,
        embedding_model=settings.embedding_model,
        created_at=manifest.get("created_at", "unknown"),
        reused_existing=True,
    )


def build_index(
    settings: Settings,
    *,
    csv_path: Path | None = None,
    max_documents: int | None = None,
    reset: bool = False,
    force: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> IndexStats:
    """Prepare the Kaggle corpus, embed its chunks and persist them in Chroma."""
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    store = VectorStore(
        settings.chroma_dir,
        settings.collection_name,
        settings.embedding_model,
        allow_model_mismatch=reset,
    )
    manifest = _read_manifest(settings)
    manifest_is_complete = (
        bool(manifest)
        and int(manifest.get("collection_count", -1)) == store.count
        and int(manifest.get("chunks", -1)) == store.count
        and manifest.get("embedding_model") == settings.embedding_model
        and int(manifest.get("chunk_size", -1)) == settings.chunk_size
        and int(manifest.get("chunk_overlap", -1)) == settings.chunk_overlap
    )
    if store.count > 0 and not reset and not force and manifest_is_complete:
        return _read_existing_stats(settings, store)
    if reset:
        store.reset()
        settings.manifest_path.unlink(missing_ok=True)

    source_file = csv_path or select_corpus_file(download_corpus())
    documents = load_corpus(source_file, max_documents=max_documents)
    chunks = list(iter_chunks(documents, settings.chunk_size, settings.chunk_overlap))
    if not chunks:
        raise ValueError("No valid chunks were produced from the corpus")

    existing_ids = store.existing_ids() if store.count and not force else set()
    expected_ids = {chunk.chunk_id for chunk in chunks}
    unexpected_ids = existing_ids.difference(expected_ids)
    if unexpected_ids:
        raise ValueError(
            f"The existing collection contains {len(unexpected_ids)} chunks from a "
            "different corpus configuration. Rebuild it with --reset."
        )
    pending_chunks = (
        chunks if force else [chunk for chunk in chunks if chunk.chunk_id not in existing_ids]
    )

    embedder = EmbeddingService(settings.embedding_model)
    processed = len(chunks) - len(pending_chunks)
    total = len(chunks)
    if progress:
        progress(processed, total)
    for batch in _batches(pending_chunks, settings.embedding_batch_size):
        batch = list(batch)
        vectors = embedder.encode(
            [chunk.embedding_text for chunk in batch],
            batch_size=settings.embedding_batch_size,
        )
        store.upsert(batch, vectors)
        processed += len(batch)
        if progress:
            progress(processed, total)

    created_at = datetime.now(timezone.utc).isoformat()
    stats = IndexStats(
        source_file=str(source_file),
        documents=len(documents),
        chunks=len(chunks),
        collection_count=store.count,
        embedding_model=settings.embedding_model,
        created_at=created_at,
    )
    manifest = {
        **asdict(stats),
        "dataset_url": DATASET_URL,
        "collection_name": settings.collection_name,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
    }
    settings.manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return stats
