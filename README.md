---
title: arXiv Evidence RAG
emoji: 🔬
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# arXiv Evidence RAG

Sistema de Recuperación Aumentada por Generación (RAG) para consultar en lenguaje natural los títulos, resúmenes y categorías del corpus público [arXiv Paper Abstracts de Kaggle](https://www.kaggle.com/datasets/spsayakpaul/arxiv-paper-abstracts).

El proyecto implementa el enunciado de `examen2bim.pdf`: preparación del corpus, embeddings, almacenamiento vectorial, recuperación, re-ranking, generación fundamentada, presentación de evidencias, chat web y configuración de despliegue. Según el alcance solicitado, la evaluación incluida es **cualitativa**; no se implementan métricas estándar de recuperación.

## Arquitectura

```text
Kaggle (CSV)
    │
    ▼
limpieza + deduplicación + fragmentación
    │
    ▼
all-MiniLM-L6-v2 ──► Chroma persistente (cosine/HNSW)
                              │ top 20
                              ▼
             ms-marco-MiniLM-L-6-v2 (re-ranking)
                              │ top 5, documentos diversos
                              ▼
       prompt con evidencias [E1]…[E5] ──► LLM
                              │
                              ▼
                  respuesta + evidencias en Streamlit
```

### Decisiones principales

- Se selecciona `arxiv_data_210930-054931.csv`, la variante más reciente del dataset y la que contiene la columna `abstracts`.
- Las filas duplicadas se consolidan y sus categorías se combinan.
- Cada abstract se divide en ventanas solapadas configurables; título y categorías también forman parte del texto usado para obtener el embedding.
- Los embeddings se normalizan y Chroma usa distancia coseno.
- El recuperador obtiene candidatos mediante búsqueda densa y un cross-encoder vuelve a ordenarlos.
- Se limita por defecto a un fragmento por paper para favorecer evidencia procedente de varios documentos.
- Si las puntuaciones no superan los umbrales configurados, el LLM no se invoca y el sistema declara que el corpus no tiene información suficiente.
- El prompt obliga al LLM a contestar en el idioma de la consulta, usar únicamente el contexto y citar afirmaciones con identificadores como `[E1]`.

## Estructura

```text
.
├── app.py                         # Chat web Streamlit
├── examen_final_rag.ipynb         # Notebook de entrega (literales A–I)
├── scripts/build_index.py         # Descarga e indexación reproducible
├── src/arxiv_rag/
│   ├── config.py                  # Variables de entorno
│   ├── corpus.py                  # Descarga, limpieza y fragmentación
│   ├── embeddings.py              # Sentence Transformer
│   ├── vector_store.py            # Persistencia y consulta Chroma
│   ├── indexing.py                # Construcción del índice
│   ├── retriever.py               # Recuperación y re-ranking
│   ├── llm.py                     # API compatible con OpenAI
│   └── rag.py                     # Orquestación RAG
├── tests/                          # Pruebas unitarias
├── Dockerfile                      # Hugging Face Spaces/Cloud Run/Render
└── .env.example                    # Configuración sin secretos
```

## Ejecución local

Se recomienda Python 3.11.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Añadir una clave en `.env`. Por defecto se usa la API compatible con OpenAI de Groq:

```dotenv
LLM_API_KEY=tu_clave
LLM_API_BASE=https://api.groq.com/openai/v1
LLM_MODEL=llama-3.3-70b-versatile
```

También puede usarse otro proveedor compatible cambiando esas tres variables. Nunca se debe subir `.env` al repositorio.

### 1. Construir el índice

Corpus completo:

```bash
python scripts/build_index.py --reset
```

Prueba rápida con una parte del corpus:

```bash
python scripts/build_index.py --max-documents 2000 --reset
```

La descarga pública se realiza con `kagglehub`; no requiere guardar el CSV en Git. Los artefactos quedan en `data/chroma/` y `data/index_manifest.json` y están excluidos por `.gitignore`.

### 2. Abrir el chat

```bash
streamlit run app.py
```

Abrir `http://localhost:7860`. El chat conserva mensajes para visualización, pero cada consulta se recupera y genera independientemente.

## Configuración relevante

| Variable | Valor por defecto | Función |
|---|---:|---|
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Modelo de embeddings |
| `RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Cross-encoder |
| `RETRIEVAL_CANDIDATES` | `20` | Candidatos densos |
| `TOP_K` | `5` | Evidencias finales |
| `CHUNK_SIZE` | `1800` | Caracteres por fragmento |
| `CHUNK_OVERLAP` | `250` | Solapamiento |
| `MIN_SEMANTIC_SCORE` | `0.20` | Umbral coseno |
| `MIN_RERANK_SCORE` | `0.05` | Umbral del cross-encoder tras sigmoid |
| `AUTO_BUILD_INDEX` | `false` | Construye el índice al iniciar si está vacío |
| `MAX_DOCUMENTS` | vacío | Vacío usa todo el corpus |

Los umbrales sirven para abstención, no como métricas de evaluación. Deben calibrarse cualitativamente con preguntas dentro y fuera del dominio.

## Despliegue en Hugging Face Spaces

1. Crear un Space nuevo y elegir **Docker** como SDK.
2. Subir este directorio al repositorio del Space.
3. En **Settings → Variables and secrets**, añadir `LLM_API_KEY` como secreto.
4. Mantener `AUTO_BUILD_INDEX=true` (ya es el valor del `Dockerfile`). Para una demostración corta puede definirse `MAX_DOCUMENTS=10000`; para la entrega final debe dejarse vacío y utilizar el corpus completo.
5. Esperar a que la primera ejecución descargue el corpus y construya Chroma. La interfaz muestra el progreso.
6. Copiar la URL pública del Space en la sección H del notebook.

Para evitar reconstrucciones en cada nuevo build, puede construirse `data/chroma` antes de crear la imagen o montar almacenamiento persistente en un proveedor como Render/Railway/Cloud Run. Los artefactos del índice no contienen claves.

### Docker local

```bash
docker build -t arxiv-rag .
docker run --rm -p 7860:7860 \
  -e LLM_API_KEY="$LLM_API_KEY" \
  -v "$PWD/data:/app/data" \
  arxiv-rag
```

El volumen conserva el índice entre reinicios.

## Evaluación cualitativa

El notebook contiene una tabla para registrar manualmente, por consulta:

1. corrección de la respuesta;
2. relevancia respecto de la consulta;
3. fidelidad a las evidencias;
4. integración de varios documentos;
5. abstención cuando el corpus es insuficiente;
6. observaciones y errores detectados.

Conviene incluir preguntas temáticas y al menos una pregunta deliberadamente ajena al corpus. Esta evaluación no calcula métricas automáticas estándar.

## Pruebas

```bash
pip install -r requirements-dev.txt
pytest -q
```

## URL de la aplicación

Reemplazar después del despliegue:

```text
https://huggingface.co/spaces/USUARIO/NOMBRE-DEL-SPACE
```
