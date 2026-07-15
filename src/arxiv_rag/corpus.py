from __future__ import annotations

import ast
import hashlib
import re
from collections import OrderedDict
from pathlib import Path
from typing import Iterable, Iterator

import kagglehub
import pandas as pd

from .models import Chunk, CorpusDocument

DATASET_HANDLE = "spsayakpaul/arxiv-paper-abstracts"
DATASET_URL = "https://www.kaggle.com/datasets/spsayakpaul/arxiv-paper-abstracts"

_WHITESPACE = re.compile(r"\s+")


def download_corpus() -> Path:
    """Download the public Kaggle dataset and return its extracted directory."""
    return Path(kagglehub.dataset_download(DATASET_HANDLE))


def select_corpus_file(dataset_dir: Path) -> Path:
    """Select the newest CSV variant, preferring the file with `abstracts`."""
    candidates = sorted(dataset_dir.glob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No CSV files found in {dataset_dir}")

    for path in sorted(candidates, key=lambda item: item.stat().st_mtime, reverse=True):
        columns = set(pd.read_csv(path, nrows=0).columns)
        if {"titles", "abstracts", "terms"}.issubset(columns):
            return path
    for path in candidates:
        columns = set(pd.read_csv(path, nrows=0).columns)
        if {"titles", "summaries", "terms"}.issubset(columns):
            return path
    raise ValueError("The Kaggle files do not have the expected title/abstract/category columns")


def clean_text(value: object) -> str:
    if value is None:
        return ""
    if not isinstance(value, (list, tuple, set, dict)) and pd.isna(value):
        return ""
    return _WHITESPACE.sub(" ", str(value)).strip()


def parse_categories(value: object) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, (list, tuple, set)) and pd.isna(value):
        return ()
    if isinstance(value, (list, tuple, set)):
        items = value
    else:
        raw = str(value).strip()
        try:
            parsed = ast.literal_eval(raw)
            items = parsed if isinstance(parsed, (list, tuple, set)) else [parsed]
        except (ValueError, SyntaxError):
            items = raw.split(",")
    return tuple(sorted({clean_text(item).strip("'\"") for item in items if clean_text(item)}))


def _document_id(title: str, abstract: str) -> str:
    digest = hashlib.sha1(f"{title}\n{abstract}".encode("utf-8")).hexdigest()[:16]
    return f"arxiv-{digest}"


def load_corpus(csv_path: Path, max_documents: int | None = None) -> list[CorpusDocument]:
    """Load, clean and deduplicate papers from the selected Kaggle CSV."""
    frame = pd.read_csv(csv_path)
    abstract_column = "abstracts" if "abstracts" in frame.columns else "summaries"
    required = {"titles", abstract_column, "terms"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {', '.join(sorted(missing))}")

    # Ordered mapping preserves the original corpus order while combining category labels
    # from duplicate rows.
    records: OrderedDict[str, dict] = OrderedDict()
    for row in frame[["titles", abstract_column, "terms"]].itertuples(index=False, name=None):
        title, abstract, terms = clean_text(row[0]), clean_text(row[1]), parse_categories(row[2])
        if not title or not abstract:
            continue
        key = hashlib.sha1(f"{title.casefold()}\n{abstract.casefold()}".encode("utf-8")).hexdigest()
        if key in records:
            records[key]["categories"].update(terms)
            continue
        records[key] = {"title": title, "abstract": abstract, "categories": set(terms)}
        if max_documents is not None and len(records) >= max_documents:
            break

    return [
        CorpusDocument(
            doc_id=_document_id(item["title"], item["abstract"]),
            title=item["title"],
            abstract=item["abstract"],
            categories=tuple(sorted(item["categories"])),
        )
        for item in records.values()
    ]


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Create readable character windows, moving boundaries to nearby whitespace."""
    text = clean_text(text)
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        proposed_end = min(start + chunk_size, len(text))
        end = proposed_end
        if proposed_end < len(text):
            boundary = text.rfind(" ", start + int(chunk_size * 0.65), proposed_end)
            if boundary > start:
                end = boundary
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        next_start = max(0, end - overlap)
        boundary = text.find(" ", next_start, end)
        start = boundary + 1 if boundary != -1 else next_start
        if start >= end:
            start = end
    return chunks


def iter_chunks(
    documents: Iterable[CorpusDocument], chunk_size: int, overlap: int
) -> Iterator[Chunk]:
    for document in documents:
        pieces = split_text(document.abstract, chunk_size, overlap)
        for index, piece in enumerate(pieces):
            yield Chunk(
                chunk_id=f"{document.doc_id}:c{index:03d}",
                doc_id=document.doc_id,
                title=document.title,
                categories=document.categories,
                text=piece,
                chunk_index=index,
                total_chunks=len(pieces),
            )
