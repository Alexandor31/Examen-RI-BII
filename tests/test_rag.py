from __future__ import annotations

from dataclasses import replace

from arxiv_rag.config import Settings
from arxiv_rag.models import Evidence, RetrievalOutcome
from arxiv_rag.rag import RAGPipeline


class FakeRetriever:
    def __init__(self, outcome: RetrievalOutcome):
        self.outcome = outcome

    def retrieve(self, query: str) -> RetrievalOutcome:
        return replace(self.outcome, query=query)


class FakeLLM:
    def __init__(self, answer: str):
        self.answer = answer
        self.calls = 0

    def generate(self, query: str, context: str) -> str:
        self.calls += 1
        assert "[E1]" in context
        return self.answer


def evidence() -> tuple[Evidence, ...]:
    return (
        Evidence(
            evidence_id="E1",
            rank=1,
            chunk_id="paper:c000",
            doc_id="paper",
            title="A grounded paper",
            categories=("cs.AI",),
            text="The paper studies retrieval augmented generation.",
            semantic_score=0.8,
            rerank_score=0.9,
        ),
    )


def test_pipeline_does_not_call_llm_when_retrieval_is_insufficient() -> None:
    settings = Settings.from_env()
    outcome = RetrievalOutcome("", evidence(), True, 2.0)
    llm = FakeLLM("must not be returned")
    pipeline = RAGPipeline(settings, FakeRetriever(outcome), llm)

    result = pipeline.answer("¿Qué dice el corpus?")

    assert result.insufficient is True
    assert result.metadata["llm_called"] is False
    assert llm.calls == 0
    assert "no contiene evidencia" in result.answer


def test_pipeline_builds_cited_context_and_generates() -> None:
    settings = Settings.from_env()
    outcome = RetrievalOutcome("", evidence(), False, 2.0)
    llm = FakeLLM("RAG uses retrieved context [E1].")
    pipeline = RAGPipeline(settings, FakeRetriever(outcome), llm)

    result = pipeline.answer("What does RAG use?")

    assert result.insufficient is False
    assert result.answer.endswith("[E1].")
    assert result.metadata["llm_called"] is True
    assert llm.calls == 1
