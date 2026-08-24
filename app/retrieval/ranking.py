"""Score normalization, hybrid scoring, and metadata-driven document precedence."""

from __future__ import annotations

from typing import Any
from app.retrieval.models import Chunk, RetrievalConfig, ScoredChunk


def calculate_authority_score(metadata: dict[str, Any]) -> float:
    """Calculate generic document authority and precedence score from front-matter metadata.

    Factors considered (strictly generic metadata, never hardcoded filenames):
    - status: active (1.0), superseded (0.15), draft (0.05)
    - policy_authority: official (1.0), none (0.0)
    - customer_answering: False (0.05), True/omitted (1.0)
    - superseded_by: present (0.15)
    - audience: customer (1.0), internal (0.7)
    """
    # 1. Lifecycle status
    status = str(metadata.get("status", "active")).lower().strip()
    if status == "active":
        status_factor = 1.0
    elif status == "superseded":
        status_factor = 0.15
    elif status == "draft":
        status_factor = 0.05
    else:
        status_factor = 0.6

    # If explicitly superseded by another doc ID, apply supersession penalty
    if metadata.get("superseded_by"):
        status_factor = min(status_factor, 0.15)

    # 2. Policy authority level
    authority = str(metadata.get("policy_authority", "official")).lower().strip()
    if authority == "official":
        auth_factor = 1.0
    elif authority == "none":
        auth_factor = 0.0
    else:
        auth_factor = 0.75

    # 3. Customer answering flag
    customer_answering = metadata.get("customer_answering", True)
    answering_factor = 1.0 if customer_answering is not False else 0.05

    # 4. Target audience
    audience = str(metadata.get("audience", "customer")).lower().strip()
    if audience == "customer":
        audience_factor = 1.0
    elif audience == "internal":
        audience_factor = 0.70
    else:
        audience_factor = 0.85

    composite_score = status_factor * auth_factor * answering_factor * audience_factor
    return max(0.0, min(1.0, float(composite_score)))


class HybridRanker:
    """Normalizes scores and computes final hybrid rankings with document precedence."""

    def __init__(self, config: RetrievalConfig) -> None:
        self.config = config

    def rank(
        self,
        semantic_results: list[tuple[Chunk, float]],
        bm25_results: list[tuple[Chunk, float]],
        top_k: int = 5,
    ) -> list[ScoredChunk]:
        """Combine lexical and semantic retrieval results, normalize scores, and apply precedence."""
        # Collect candidate chunks keyed by chunk_id
        candidates: dict[str, dict[str, Any]] = {}

        # Process semantic results (scores are cosine similarity [0, 1])
        for chunk, sem_score in semantic_results:
            candidates[chunk.chunk_id] = {
                "chunk": chunk,
                "semantic_score": max(0.0, float(sem_score)),
                "bm25_score": 0.0,
            }

        # Process BM25 results
        for chunk, bm25_score in bm25_results:
            if chunk.chunk_id not in candidates:
                candidates[chunk.chunk_id] = {
                    "chunk": chunk,
                    "semantic_score": 0.0,
                    "bm25_score": max(0.0, float(bm25_score)),
                }
            else:
                candidates[chunk.chunk_id]["bm25_score"] = max(0.0, float(bm25_score))

        if not candidates:
            return []

        # Use absolute reference for BM25 score normalization to avoid inflating noise
        # In BM25Okapi over short passages, a solid match has score ~8.0-15.0+
        bm25_scale_factor = max(10.0, max((data["bm25_score"] for data in candidates.values()), default=10.0))

        scored_chunks: list[ScoredChunk] = []

        sem_w = self.config.semantic_weight
        bm25_w = self.config.bm25_weight
        auth_w = self.config.authority_weight
        total_rel_w = sem_w + bm25_w if (sem_w + bm25_w) > 0 else 1.0

        for chunk_id, data in candidates.items():
            chunk: Chunk = data["chunk"]
            raw_sem: float = data["semantic_score"]
            raw_bm25: float = data["bm25_score"]

            # Normalized semantic similarity [0, 1]
            norm_sem = min(1.0, max(0.0, raw_sem))

            # Normalized BM25 lexical score [0, 1] scaled against reference threshold
            norm_bm25 = min(1.0, max(0.0, raw_bm25 / bm25_scale_factor))
            # If raw BM25 is very weak (under 2.5), discount it as lexical noise
            if raw_bm25 < 2.5:
                norm_bm25 = norm_bm25 * (raw_bm25 / 2.5)

            # Metadata authority score
            authority_score = calculate_authority_score(chunk.metadata)
            status = str(chunk.metadata.get("status", "active"))

            # Normalized base relevance
            relevance = (sem_w * norm_sem + bm25_w * norm_bm25) / total_rel_w

            # Authority gating: draft/superseded/none docs are discounted
            authority_multiplier = 0.20 + 0.80 * authority_score
            final_score = relevance * authority_multiplier

            # Blend with authority weight
            if auth_w > 0:
                final_score = (1.0 - auth_w) * final_score + auth_w * (authority_score * relevance)

            final_score = float(max(0.0, min(1.0, final_score)))

            scored_chunks.append(
                ScoredChunk(
                    chunk=chunk,
                    semantic_score=round(norm_sem, 4),
                    bm25_score=round(raw_bm25, 4),
                    authority_score=round(authority_score, 4),
                    final_score=round(final_score, 4),
                    status=status,
                )
            )

        # Sort descending by final_score, then by authority_score, then by semantic_score
        scored_chunks.sort(
            key=lambda x: (x.final_score, x.authority_score, x.semantic_score, x.bm25_score),
            reverse=True,
        )

        return scored_chunks[:top_k]
