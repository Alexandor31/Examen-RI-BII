from __future__ import annotations

import re
import time

from .config import Settings
from .llm import OpenAICompatibleLLM
from .models import Evidence, RAGResult
from .retriever import Retriever

_SPANISH_HINTS = re.compile(
    r"\b(qué|cómo|cuál|cuáles|dónde|por qué|sobre|artículos|investigación|modelos|aprendizaje)\b",
    flags=re.IGNORECASE,
)


class RAGPipeline:
    def __init__(
        self,
        settings: Settings,
        retriever: Retriever | None = None,
        llm: OpenAICompatibleLLM | None = None,
    ):
        self.settings = settings
        self.retriever = retriever or Retriever(settings)
        self.llm = llm or OpenAICompatibleLLM(settings)

    def _insufficient_answer(self, query: str) -> str:
        if _SPANISH_HINTS.search(query):
            return (
                "El corpus no contiene evidencia suficientemente relevante para responder "
                "esta consulta con confianza. Intenta reformularla o preguntar sobre otro "
                "tema presente en los resúmenes de arXiv."
            )
        return (
            "The corpus does not contain sufficiently relevant evidence to answer this "
            "question confidently. Try rephrasing it or asking about another topic covered "
            "by the arXiv abstracts."
        )

    def _build_context(self, evidence: tuple[Evidence, ...]) -> str:
        blocks: list[str] = []
        used = 0
        for item in evidence:
            block = (
                f"[{item.evidence_id}]\n"
                f"Title: {item.title}\n"
                f"Categories: {', '.join(item.categories) or 'uncategorized'}\n"
                f"Abstract fragment: {item.text}"
            )
            if blocks and used + len(block) > self.settings.max_context_chars:
                break
            blocks.append(block)
            used += len(block)
        return "\n\n".join(blocks)

    def answer(self, query: str) -> RAGResult:
        retrieval = self.retriever.retrieve(query)
        if retrieval.insufficient:
            return RAGResult(
                query=retrieval.query,
                answer=self._insufficient_answer(retrieval.query),
                evidence=retrieval.evidence,
                insufficient=True,
                retrieval_ms=retrieval.retrieval_ms,
                generation_ms=0.0,
                warning=retrieval.warning,
                metadata={"llm_called": False},
            )

        context = self._build_context(retrieval.evidence)
        started = time.perf_counter()
        answer = self.llm.generate(retrieval.query, context)
        generation_ms = (time.perf_counter() - started) * 1000
        insufficient_marker = answer.startswith("[INSUFFICIENT_CONTEXT]")
        if insufficient_marker:
            answer = answer.removeprefix("[INSUFFICIENT_CONTEXT]").strip()
            if not answer:
                answer = self._insufficient_answer(retrieval.query)

        return RAGResult(
            query=retrieval.query,
            answer=answer,
            evidence=retrieval.evidence,
            insufficient=insufficient_marker,
            retrieval_ms=retrieval.retrieval_ms,
            generation_ms=generation_ms,
            warning=retrieval.warning,
            metadata={
                "llm_called": True,
                "model": self.settings.llm_model,
                "evidence_count": len(retrieval.evidence),
            },
        )
