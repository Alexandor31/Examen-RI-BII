from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class CorpusDocument:
    doc_id: str
    title: str
    abstract: str
    categories: tuple[str, ...]


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    title: str
    categories: tuple[str, ...]
    text: str
    chunk_index: int
    total_chunks: int

    @property
    def embedding_text(self) -> str:
        categories = ", ".join(self.categories) or "uncategorized"
        return (
            f"Title: {self.title}\n"
            f"Categories: {categories}\n"
            f"Abstract fragment: {self.text}"
        )

    @property
    def metadata(self) -> dict[str, str | int]:
        return {
            "doc_id": self.doc_id,
            "title": self.title,
            "categories": ", ".join(self.categories),
            "chunk_index": self.chunk_index,
            "total_chunks": self.total_chunks,
        }


@dataclass(frozen=True)
class Evidence:
    evidence_id: str
    rank: int
    chunk_id: str
    doc_id: str
    title: str
    categories: tuple[str, ...]
    text: str
    semantic_score: float
    rerank_score: float | None = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class RetrievalOutcome:
    query: str
    evidence: tuple[Evidence, ...]
    insufficient: bool
    retrieval_ms: float
    warning: str | None = None


@dataclass(frozen=True)
class RAGResult:
    query: str
    answer: str
    evidence: tuple[Evidence, ...]
    insufficient: bool
    retrieval_ms: float
    generation_ms: float
    warning: str | None = None
    metadata: dict = field(default_factory=dict)
