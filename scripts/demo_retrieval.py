"""Demonstration script for RAG retrieval traces on representative queries."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.retrieval.index import KnowledgeIndex
from app.retrieval.retriever import HybridRetriever


def main():
    storage_dir = _PROJECT_ROOT / "storage"
    if not storage_dir.exists() or not (storage_dir / "chunks.json").exists():
        print("[!] Storage index not found. Building index first...")
        from scripts.build_index import build_index
        build_index(_PROJECT_ROOT / "knowledge-base", storage_dir, verbose=False)

    index = KnowledgeIndex.load(storage_dir)
    retriever = HybridRetriever(index)

    queries = [
        ("Return Policy Question", "What is the return window for a standard customer?"),
        ("International Shipping Question", "Do you ship internationally to Canada?"),
        ("Product Care Question", "How do I care for and clean fabric bags and backpacks?"),
        ("Insufficient Evidence Question", "Are all fabrics and adhesives in your bags vegan?"),
        ("Source Conflict Question", "Can I put the entire Breeze Tumbler in the dishwasher?"),
    ]

    print("=" * 80)
    print("ASTER & ROW SUPPORT AGENT - RAG SUBSYSTEM RETRIEVAL TRACES")
    print("=" * 80)
    print()

    for category, query in queries:
        print(f"### CATEGORY: {category}")
        result = retriever.search(query, top_k=3)
        print(result.debug_trace)
        print("-" * 80)
        print()


if __name__ == "__main__":
    main()
