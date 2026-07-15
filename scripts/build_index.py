#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from arxiv_rag.config import Settings
from arxiv_rag.indexing import build_index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download, clean, embed and index the Kaggle arXiv corpus in Chroma."
    )
    parser.add_argument(
        "--data-file",
        type=Path,
        help="Optional local Kaggle CSV; otherwise it is downloaded with kagglehub.",
    )
    parser.add_argument(
        "--max-documents",
        type=int,
        default=None,
        help="Limit unique papers for a quick demo. Omit it to use the complete corpus.",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="Delete the collection before rebuilding it.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Upsert the corpus even if a non-empty index already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = Settings.from_env()

    def progress(done: int, total: int) -> None:
        percentage = done * 100 / total
        print(f"\rIndexing: {done:,}/{total:,} ({percentage:5.1f}%)", end="", flush=True)

    stats = build_index(
        settings,
        csv_path=args.data_file,
        max_documents=args.max_documents,
        reset=args.reset,
        force=args.force,
        progress=progress,
    )
    if not stats.reused_existing:
        print()
    print(f"Source:      {stats.source_file}")
    print(f"Documents:   {stats.documents:,}")
    print(f"Chunks:      {stats.chunks:,}")
    print(f"Chroma rows: {stats.collection_count:,}")
    print(f"Model:       {stats.embedding_model}")
    if stats.reused_existing:
        print("The existing index was reused. Pass --reset to rebuild it.")


if __name__ == "__main__":
    main()
