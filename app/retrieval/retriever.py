"""Hybrid Knowledge Base Retriever with precedence, conflict detection, and sufficiency analysis."""

from __future__ import annotations

from typing import Optional
from app.retrieval.conflicts import ConflictDetector
from app.retrieval.index import KnowledgeIndex
from app.retrieval.models import RetrievalConfig, RetrievalResult, ScoredChunk
from app.retrieval.ranking import HybridRanker


class HybridRetriever:
    """Hybrid BM25 + FAISS Retriever with authority ranking, conflict detection, and sufficiency gating."""

    def __init__(
        self,
        index: KnowledgeIndex,
        config: Optional[RetrievalConfig] = None,
        conflict_detector: Optional[ConflictDetector] = None,
    ) -> None:
        self.index = index
        self.config = config or RetrievalConfig()
        self.ranker = HybridRanker(self.config)
        self.conflict_detector = conflict_detector or ConflictDetector()

    def search(self, query: str, top_k: Optional[int] = None) -> RetrievalResult:
        """Execute hybrid search over the knowledge base.

        Args:
            query: User's natural language question or search query.
            top_k: Number of top results to return (defaults to config.top_k).

        Returns:
            RetrievalResult containing ranked ScoredChunks, conflict flags, sufficiency flags, and debug trace.
        """
        k = top_k or self.config.top_k
        clean_query = query.strip()

        if not clean_query:
            return RetrievalResult(
                query=query,
                chunks=[],
                conflict=False,
                conflict_sources=[],
                insufficient=True,
                debug_trace="Empty query provided.",
            )

        # 1. Retrieve candidates
        # Fetch 2x top_k from each source for a diverse candidate pool
        fetch_k = max(10, k * 2)
        semantic_results = self.index.search_semantic(clean_query, top_k=fetch_k)
        bm25_results = self.index.search_bm25(clean_query, top_k=fetch_k)

        # 2. Rank and apply document precedence
        scored_chunks = self.ranker.rank(
            semantic_results=semantic_results,
            bm25_results=bm25_results,
            top_k=fetch_k,
        )

        # 3. Check for insufficient evidence
        insufficient = self._check_insufficient(scored_chunks)

        # 4. Check for active source conflicts
        conflict, conflict_sources, conflict_details = self.conflict_detector.detect_conflicts(
            query=clean_query,
            scored_chunks=scored_chunks,
        )

        # 5. Select final top_k chunks (ensuring all conflicting chunks are included if present)
        final_chunks = self._select_final_chunks(scored_chunks, conflict_sources, k)

        # 6. Generate human-readable debug trace
        debug_trace = self._generate_debug_trace(
            query=clean_query,
            chunks=final_chunks,
            conflict=conflict,
            conflict_sources=conflict_sources,
            conflict_details=conflict_details,
            insufficient=insufficient,
        )

        return RetrievalResult(
            query=query,
            chunks=final_chunks,
            conflict=conflict,
            conflict_sources=conflict_sources,
            conflict_details=conflict_details,
            insufficient=insufficient,
            debug_trace=debug_trace,
        )

    def _check_insufficient(self, scored_chunks: list[ScoredChunk]) -> bool:
        """Determine if retrieved evidence is insufficient to reliably answer the query."""
        if not scored_chunks:
            return True

        top_chunk = scored_chunks[0]

        # If top chunk has zero or negligible relevance
        if top_chunk.final_score < 0.10:
            return True

        # If top chunk has low overall final score and lacks dual semantic/lexical support
        if top_chunk.final_score < self.config.insufficient_similarity_threshold:
            if top_chunk.semantic_score < self.config.insufficient_semantic_threshold or top_chunk.bm25_score < 2.0:
                return True

        # If top chunk is draft / non-authoritative
        if top_chunk.authority_score <= 0.1:
            return True

        return False

    def _select_final_chunks(
        self,
        scored_chunks: list[ScoredChunk],
        conflict_sources: list[str],
        top_k: int,
    ) -> list[ScoredChunk]:
        """Select top_k chunks while making sure conflicting sources are both represented."""
        selected: list[ScoredChunk] = []
        selected_files: set[str] = set()

        # If conflict sources exist, ensure at least one top chunk from each conflicting file is included
        if conflict_sources:
            for conf_file in conflict_sources:
                for sc in scored_chunks:
                    if sc.filename == conf_file and sc not in selected:
                        selected.append(sc)
                        selected_files.add(sc.filename)
                        break

        # Fill remaining slots up to top_k
        for sc in scored_chunks:
            if len(selected) >= top_k:
                break
            if sc not in selected:
                selected.append(sc)

        return selected

    def _generate_debug_trace(
        self,
        query: str,
        chunks: list[ScoredChunk],
        conflict: bool,
        conflict_sources: list[str],
        conflict_details: Optional[str],
        insufficient: bool,
    ) -> str:
        """Generate structured debug output explaining retrieval decisions."""
        lines = [f'Query: "{query}"', ""]

        if insufficient:
            lines.append("Decision: INSUFFICIENT INFORMATION (confidence below threshold)")
            lines.append("")

        if conflict:
            lines.append(f"Decision: CONFLICT DETECTED between sources {conflict_sources}")
            if conflict_details:
                lines.append(f"Details: {conflict_details}")
            lines.append("")

        lines.append(f"Retrieved {len(chunks)} passage(s):")
        for i, sc in enumerate(chunks, 1):
            lines.append(f"{i}. {sc.filename}")
            lines.append(f"   Heading: {sc.heading}")
            lines.append(f"   Heading Path: {sc.heading_path}")
            lines.append(f"   Semantic: {sc.semantic_score:.4f}")
            lines.append(f"   BM25: {sc.bm25_score:.4f}")
            lines.append(f"   Authority: {sc.authority_score:.4f}")
            lines.append(f"   Final: {sc.final_score:.4f}")
            lines.append(f"   Status: {sc.status}")
            lines.append(f"   Source Citation: Source: {sc.source_reference}")
            lines.append("")

        return "\n".join(lines).strip()
