from __future__ import annotations

from pathlib import Path

import pandas as pd

from arxiv_rag.corpus import load_corpus, parse_categories, split_text


def test_parse_categories_handles_kaggle_format() -> None:
    assert parse_categories("['cs.LG', 'cs.AI', 'cs.LG']") == ("cs.AI", "cs.LG")
    assert parse_categories("cs.CV, stat.ML") == ("cs.CV", "stat.ML")
    assert parse_categories(["cs.LG", "cs.AI"]) == ("cs.AI", "cs.LG")


def test_load_corpus_cleans_deduplicates_and_merges_categories(tmp_path: Path) -> None:
    csv_path = tmp_path / "papers.csv"
    pd.DataFrame(
        {
            "titles": [" A  paper ", "A paper", "Second paper", None],
            "abstracts": [
                "An abstract\nwith spacing.",
                "An abstract with spacing.",
                "Another abstract.",
                "Ignored abstract.",
            ],
            "terms": ["['cs.LG']", "['cs.AI']", "['cs.CV']", "['cs.AI']"],
        }
    ).to_csv(csv_path, index=False)

    documents = load_corpus(csv_path)

    assert len(documents) == 2
    assert documents[0].title == "A paper"
    assert documents[0].abstract == "An abstract with spacing."
    assert documents[0].categories == ("cs.AI", "cs.LG")
    assert documents[0].doc_id.startswith("arxiv-")


def test_split_text_returns_overlapping_bounded_chunks() -> None:
    text = " ".join(f"word{i}" for i in range(150))
    chunks = split_text(text, chunk_size=180, overlap=30)

    assert len(chunks) > 1
    assert all(chunk.strip() == chunk for chunk in chunks)
    assert all(len(chunk) <= 180 for chunk in chunks)
    assert "word0" in chunks[0]
    assert "word149" in chunks[-1]
