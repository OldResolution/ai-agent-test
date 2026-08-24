"""Build and persist BM25 and FAISS indices from the knowledge base."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Ensure project root is in sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from app.retrieval.chunker import MarkdownChunker
from app.retrieval.embeddings import get_embedding_provider
from app.retrieval.index import KnowledgeIndex
from app.retrieval.loader import MarkdownLoader


def build_index(
    kb_dir: Path,
    storage_dir: Path,
    use_mock: bool = False,
    verbose: bool = True,
) -> KnowledgeIndex:
    """Read knowledge base markdown files, chunk them, build BM25 & FAISS index, and save to storage."""
    load_dotenv()

    if verbose:
        print(f"[*] Scanning knowledge base directory: {kb_dir}")

    loader = MarkdownLoader(base_dir=kb_dir)
    documents = loader.load_directory()

    if not documents:
        raise ValueError(f"No markdown documents found in {kb_dir}")

    if verbose:
        print(f"[+] Loaded {len(documents)} document(s).")
        for doc in documents:
            status = doc.metadata.get("status", "unknown")
            authority = doc.metadata.get("policy_authority", "unknown")
            print(f"    - {doc.filename} (ID: {doc.document_id}, status: {status}, authority: {authority})")

    # Chunk documents
    chunker = MarkdownChunker()
    chunks = chunker.chunk_documents(documents)

    if verbose:
        print(f"[+] Generated {len(chunks)} semantic chunk(s).")

    # Initialize embedding provider
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key and not use_mock:
        if verbose:
            print("[!] OPENAI_API_KEY not found in environment. Falling back to deterministic embedding provider.")
        use_mock = True

    provider = get_embedding_provider(api_key=api_key, mock=use_mock)

    if verbose:
        provider_name = provider.__class__.__name__
        print(f"[*] Building FAISS vector index & BM25 lexical index using {provider_name} (dim: {provider.dimension})...")

    index = KnowledgeIndex.build(chunks=chunks, embedding_provider=provider)

    if verbose:
        print(f"[*] Persisting index artifacts to {storage_dir}...")

    index.save(storage_dir)

    if verbose:
        print("[+] Index build complete successfully!")
        print(f"    - Chunks manifest: {storage_dir / 'chunks.json'}")
        print(f"    - FAISS vector index: {storage_dir / 'faiss.index'}")
        print(f"    - BM25 model: {storage_dir / 'bm25.pkl'}")
        print(f"    - Manifest: {storage_dir / 'manifest.json'}")

    return index


def main() -> None:
    parser = argparse.ArgumentParser(description="Build RAG search index from Markdown knowledge base.")
    parser.add_argument(
        "--kb-dir",
        type=Path,
        default=_PROJECT_ROOT / "knowledge-base",
        help="Path to the knowledge-base directory",
    )
    parser.add_argument(
        "--storage-dir",
        type=Path,
        default=_PROJECT_ROOT / "storage",
        help="Path to the storage directory for output indexes",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Force use of deterministic local embeddings (offline mode without OpenAI API key)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed progress output",
    )

    args = parser.parse_args()

    try:
        build_index(
            kb_dir=args.kb_dir,
            storage_dir=args.storage_dir,
            use_mock=args.mock,
            verbose=not args.quiet,
        )
    except Exception as e:
        print(f"[!] Error building index: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
