from __future__ import annotations

import html
import json
import os
import sys
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from arxiv_rag.config import Settings
from arxiv_rag.indexing import build_index
from arxiv_rag.llm import LLMConfigurationError, LLMRequestError
from arxiv_rag.models import Evidence, RAGResult
from arxiv_rag.rag import RAGPipeline
from arxiv_rag.vector_store import VectorStore

st.set_page_config(
    page_title="arXiv Evidence RAG",
    page_icon="🔬",
    layout="wide",
)

st.markdown(
    """
<style>
.block-container { max-width: 1100px; padding-top: 2rem; }
.evidence-card { border-left: 4px solid #5b8def; padding: .25rem 1rem; margin: .7rem 0; }
.small-muted { color: #718096; font-size: .84rem; }
[data-testid="stChatMessage"] { border: 1px solid rgba(128,128,128,.18); }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_pipeline() -> RAGPipeline:
    return RAGPipeline(Settings.from_env())


def index_count(settings: Settings) -> int:
    store = VectorStore(
        settings.chroma_dir, settings.collection_name, settings.embedding_model
    )
    return store.count


def render_evidence(evidence: tuple[Evidence, ...]) -> None:
    if not evidence:
        st.info("No se recuperaron evidencias.")
        return
    with st.expander(f"Evidencias recuperadas ({len(evidence)})", expanded=True):
        for item in evidence:
            categories = ", ".join(item.categories) or "sin categoría"
            rerank = (
                f" · re-ranking: {item.rerank_score:.3f}"
                if item.rerank_score is not None
                else ""
            )
            safe_title = html.escape(item.title)
            safe_categories = html.escape(categories)
            st.markdown(
                f"<div class='evidence-card'><strong>[{item.evidence_id}] "
                f"{safe_title}</strong><br><span class='small-muted'>"
                f"{safe_categories} · similitud semántica: {item.semantic_score:.3f}"
                f"{rerank}</span></div>",
                unsafe_allow_html=True,
            )
            st.write(item.text)


def render_result(result: RAGResult) -> None:
    if result.insufficient:
        st.warning(result.answer)
    else:
        st.markdown(result.answer)
    if result.warning:
        st.caption(f"⚠️ {result.warning}")
    st.caption(
        f"Recuperación: {result.retrieval_ms / 1000:.2f} s · "
        f"Generación: {result.generation_ms / 1000:.2f} s"
    )
    render_evidence(result.evidence)


def load_streamlit_secrets() -> None:
    """Expose Streamlit Cloud secrets through the existing environment config."""
    try:
        secrets = st.secrets
    except Exception:
        # Local runs without .streamlit/secrets.toml do not have Streamlit secrets.
        return

    names = (
        "LLM_API_KEY",
        "GROQ_API_KEY",
        "LLM_API_BASE",
        "LLM_MODEL",
        "MAX_DOCUMENTS",
        "AUTO_BUILD_INDEX",
    )
    for name in names:
        if os.getenv(name):
            continue
        try:
            value = secrets[name]
        except KeyError:
            continue
        if value is not None:
            os.environ[name] = str(value)


load_streamlit_secrets()
settings = Settings.from_env()

with st.sidebar:
    st.header("Estado del sistema")
    try:
        count = index_count(settings)
    except Exception as exc:
        st.error(f"No se pudo abrir Chroma: {exc}")
        count = 0
    if count:
        st.success(f"Índice listo: {count:,} fragmentos")
    else:
        st.warning("El índice vectorial todavía está vacío.")
    st.caption(f"Embeddings: `{settings.embedding_model.split('/')[-1]}`")
    reranker_label = settings.reranker_model.split("/")[-1] if settings.enable_reranker else "desactivado"
    st.caption(f"Re-ranking: `{reranker_label}`")
    st.caption(f"LLM: `{settings.llm_model}`")
    if settings.llm_api_key:
        st.success("Clave del LLM configurada")
    else:
        st.error("Falta el secreto `LLM_API_KEY` o `GROQ_API_KEY`.")

    if settings.manifest_path.exists():
        with st.expander("Manifiesto del índice"):
            st.json(json.loads(settings.manifest_path.read_text(encoding="utf-8")))

    if not count and st.button("Construir índice ahora", use_container_width=True):
        progress_bar = st.progress(0.0, text="Preparando el corpus…")

        def update_progress(done: int, total: int) -> None:
            progress_bar.progress(done / total, text=f"Indexando {done:,}/{total:,} fragmentos")

        try:
            stats = build_index(
                settings,
                max_documents=settings.max_documents,
                progress=update_progress,
            )
            progress_bar.progress(1.0, text="Índice terminado")
            st.success(f"Se indexaron {stats.collection_count:,} fragmentos.")
            load_pipeline.clear()
            st.rerun()
        except Exception as exc:
            st.exception(exc)

if not count and settings.auto_build_index:
    with st.status("Preparando automáticamente el corpus y el índice…", expanded=True) as status:
        progress_text = st.empty()

        def auto_progress(done: int, total: int) -> None:
            progress_text.write(f"Embeddings: {done:,}/{total:,} fragmentos")

        try:
            stats = build_index(
                settings,
                max_documents=settings.max_documents,
                progress=auto_progress,
            )
            status.update(label=f"Índice listo ({stats.collection_count:,} fragmentos)", state="complete")
            load_pipeline.clear()
            st.rerun()
        except Exception as exc:
            status.update(label="No se pudo construir el índice", state="error")
            st.exception(exc)

st.title("🔬 arXiv Evidence RAG")
st.write(
    "Consulta resúmenes científicos de arXiv mediante búsqueda semántica, "
    "re-ranking y respuestas respaldadas por evidencias verificables."
)
st.caption(
    "Cada consulta se procesa de forma independiente. El historial se conserva únicamente "
    "para visualizar la sesión actual."
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if message["role"] == "user":
            st.markdown(message["content"])
        else:
            render_result(message["result"])

query = st.chat_input(
    "Ej.: How is reinforcement learning used in robotics?",
    disabled=count == 0,
)
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)
    with st.chat_message("assistant"):
        try:
            with st.spinner("Recuperando y contrastando evidencias…"):
                result = load_pipeline().answer(query)
            render_result(result)
            st.session_state.messages.append({"role": "assistant", "result": result})
        except (LLMConfigurationError, LLMRequestError, ValueError) as exc:
            st.error(str(exc))
        except Exception as exc:
            st.error("Ocurrió un error inesperado al procesar la consulta.")
            st.exception(exc)
