"""BM25 lexical index and FAISS vector index management and persistence."""

from __future__ import annotations

import json
import os
import pickle
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Union
import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from app.retrieval.embeddings import EmbeddingProvider, get_embedding_provider
from app.retrieval.models import Chunk


_STOPWORDS = {
    "a", "an", "the", "in", "on", "at", "to", "for", "of", "with", "by", "from",
    "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "can", "could", "shall", "should", "will", "would",
    "and", "or", "but", "if", "because", "as", "what", "which", "this", "that",
    "these", "those", "then", "so", "than", "too", "very", "all", "any", "both",
    "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
    "only", "own", "same", "your", "my", "our", "their", "its", "you", "i", "we", "they"
}


def tokenize_for_bm25(text: str) -> list[str]:
    """Tokenize text into lowercase alphanumeric tokens, filtering common stopwords."""
    clean_text = text.lower()
    raw_tokens = re.findall(r"\b[a-z0-9_-]+\b", clean_text)
    meaningful = [t for t in raw_tokens if t not in _STOPWORDS and len(t) > 1]
    return meaningful if meaningful else raw_tokens


class KnowledgeIndex:
    """Combines FAISS vector index, BM25 lexical index, and chunk metadata storage."""

    def __init__(
        self,
        chunks: Optional[list[Chunk]] = None,
        faiss_index: Optional[faiss.Index] = None,
        bm25: Optional[BM25Okapi] = None,
        embedding_provider: Optional[EmbeddingProvider] = None,
    ) -> None:
        self.chunks: list[Chunk] = chunks or []
        self.faiss_index: Optional[faiss.Index] = faiss_index
        self.bm25: Optional[BM25Okapi] = bm25
        self.embedding_provider: EmbeddingProvider = (
            embedding_provider or get_embedding_provider()
        )

    @classmethod
    def build(
        cls,
        chunks: list[Chunk],
        embedding_provider: Optional[EmbeddingProvider] = None,
    ) -> KnowledgeIndex:
        """Build FAISS vector index and BM25 index from a list of chunks."""
        if not chunks:
            raise ValueError("Cannot build index from empty chunk list.")

        provider = embedding_provider or get_embedding_provider()

        # 1. Build BM25 Index
        corpus_texts = []
        tokenized_corpus = []
        for c in chunks:
            # Include heading and heading path in BM25 text representation for strong lexical grounding
            enriched_text = f"{c.heading_path}\n{c.text}"
            corpus_texts.append(enriched_text)
            tokenized_corpus.append(tokenize_for_bm25(enriched_text))

        bm25 = BM25Okapi(tokenized_corpus)

        # 2. Build FAISS Vector Index
        # Enrich text for semantic embedding with document title/heading hierarchy
        embed_texts = [f"Section: {c.heading_path}\n\n{c.text}" for c in chunks]
        embeddings = provider.embed_texts(embed_texts)
        vectors = np.array(embeddings, dtype=np.float32)

        # Ensure L2 normalization for Inner Product (cosine similarity)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized_vectors = vectors / norms

        dim = provider.dimension
        faiss_index = faiss.IndexFlatIP(dim)
        faiss_index.add(normalized_vectors)

        return cls(
            chunks=chunks,
            faiss_index=faiss_index,
            bm25=bm25,
            embedding_provider=provider,
        )

    def search_semantic(self, query: str, top_k: int = 10) -> list[tuple[Chunk, float]]:
        """Search vector index using cosine similarity."""
        if self.faiss_index is None or not self.chunks:
            return []

        query_vec = np.array([self.embedding_provider.embed_query(query)], dtype=np.float32)
        norm = np.linalg.norm(query_vec)
        if norm > 0:
            query_vec = query_vec / norm

        k = min(top_k, len(self.chunks))
        scores, indices = self.faiss_index.search(query_vec, k)

        results: list[tuple[Chunk, float]] = []
        for score, idx in zip(scores[0], indices[0]):
            if idx >= 0 and idx < len(self.chunks):
                # Clamp score to [-1.0, 1.0] and map to [0, 1]
                clamped_score = float(max(0.0, min(1.0, (score + 1.0) / 2.0)))
                # Or keep raw cosine similarity if preferred; let's preserve cosine similarity in [0, 1]
                # For non-negative similarity (or standard normalized embeddings):
                sim = float(max(0.0, min(1.0, score)))
                results.append((self.chunks[idx], sim))

        return results

    def search_bm25(self, query: str, top_k: int = 10) -> list[tuple[Chunk, float]]:
        """Search BM25 lexical index."""
        if self.bm25 is None or not self.chunks:
            return []

        query_tokens = tokenize_for_bm25(query)
        if not query_tokens:
            return [(c, 0.0) for c in self.chunks[:top_k]]

        doc_scores = self.bm25.get_scores(query_tokens)
        top_indices = np.argsort(doc_scores)[::-1][:top_k]

        results: list[tuple[Chunk, float]] = []
        for idx in top_indices:
            score = float(doc_scores[idx])
            results.append((self.chunks[idx], score))

        return results

    def save(self, storage_dir: Union[str, Path]) -> None:
        """Persist chunks, FAISS index, and BM25 index to storage directory."""
        out_dir = Path(storage_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save chunks metadata
        chunks_file = out_dir / "chunks.json"
        chunks_data = [c.model_dump(mode="json") for c in self.chunks]
        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump(chunks_data, f, indent=2, ensure_ascii=False, default=str)

        # 2. Save FAISS index
        faiss_file = out_dir / "faiss.index"
        if self.faiss_index is not None:
            faiss.write_index(self.faiss_index, str(faiss_file))

        # 3. Save BM25 model
        bm25_file = out_dir / "bm25.pkl"
        with open(bm25_file, "wb") as f:
            pickle.dump(self.bm25, f)

        # 4. Save metadata manifest
        manifest_file = out_dir / "manifest.json"
        manifest_data = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "chunk_count": len(self.chunks),
            "dimension": self.embedding_provider.dimension,
            "provider_type": self.embedding_provider.__class__.__name__,
        }
        with open(manifest_file, "w", encoding="utf-8") as f:
            json.dump(manifest_data, f, indent=2)

    @classmethod
    def load(
        cls,
        storage_dir: Union[str, Path],
        embedding_provider: Optional[EmbeddingProvider] = None,
    ) -> KnowledgeIndex:
        """Load persisted index from storage directory."""
        in_dir = Path(storage_dir)
        if not in_dir.is_dir():
            raise FileNotFoundError(f"Storage directory does not exist: {in_dir}")

        chunks_file = in_dir / "chunks.json"
        faiss_file = in_dir / "faiss.index"
        bm25_file = in_dir / "bm25.pkl"

        if not chunks_file.exists():
            raise FileNotFoundError(f"Missing chunks.json in {in_dir}")

        # 1. Load chunks
        with open(chunks_file, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)
        chunks = [Chunk.model_validate(item) for item in chunks_data]

        # 2. Load FAISS index
        faiss_index = None
        if faiss_file.exists():
            faiss_index = faiss.read_index(str(faiss_file))

        # 3. Load BM25 model
        bm25 = None
        if bm25_file.exists():
            with open(bm25_file, "rb") as f:
                bm25 = pickle.load(f)

        provider = embedding_provider or get_embedding_provider()

        return cls(
            chunks=chunks,
            faiss_index=faiss_index,
            bm25=bm25,
            embedding_provider=provider,
        )
