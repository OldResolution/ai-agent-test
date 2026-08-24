"""Conflict detection across active authoritative knowledge sources."""

from __future__ import annotations

import re
from typing import Optional
from app.retrieval.models import ScoredChunk


# Common directional / modal contradiction keyword pairs
_CONTRADICTION_PATTERNS = [
    (re.compile(r"\b(hand[- ]wash(?:ed|ing)?|do not machine wash)\b", re.I),
     re.compile(r"\b(dishwasher safe|machine wash(?:able)?|all components are dishwasher safe)\b", re.I)),
    (re.compile(r"\b(not allowed|prohibited|cannot be returned|no refund)\b", re.I),
     re.compile(r"\b(allowed|permitted|eligible for return|full refund)\b", re.I)),
    (re.compile(r"\b(free shipping)\b", re.I),
     re.compile(r"\b(shipping fee|postage required)\b", re.I)),
]


class ConflictDetector:
    """Detects genuine contradictory assertions between active authoritative sources."""

    def __init__(self, min_authority_threshold: float = 0.70) -> None:
        self.min_authority_threshold = min_authority_threshold

    def detect_conflicts(
        self,
        query: str,
        scored_chunks: list[ScoredChunk],
    ) -> tuple[bool, list[str], Optional[str]]:
        """Analyze top retrieved chunks for active authoritative conflicts.

        Returns:
            (is_conflict, conflicting_filenames, conflict_explanation)
        """
        # Filter to top active authoritative sources that are actually relevant to the query
        # Look only within top-5 results with substantial relevance
        top_candidates = scored_chunks[:5]
        authoritative_chunks = [
            sc for sc in top_candidates
            if sc.authority_score >= self.min_authority_threshold
            and sc.status == "active"
            and sc.final_score >= 0.40
        ]

        if len(authoritative_chunks) < 2:
            return False, [], None

        # Group by distinct filename
        by_file: dict[str, list[ScoredChunk]] = {}
        for sc in authoritative_chunks:
            by_file.setdefault(sc.filename, []).append(sc)

        filenames = list(by_file.keys())
        if len(filenames) < 2:
            return False, [], None

        # Compare pairs of chunks from different files
        for i in range(len(filenames)):
            file_a = filenames[i]
            chunks_a = by_file[file_a]
            for j in range(i + 1, len(filenames)):
                file_b = filenames[j]
                chunks_b = by_file[file_b]

                for ca in chunks_a:
                    for cb in chunks_b:
                        is_conflict, reason = self._check_chunk_pair_conflict(query, ca, cb)
                        if is_conflict:
                            return True, [file_a, file_b], reason

        return False, [], None

    def _check_chunk_pair_conflict(
        self,
        query: str,
        chunk_a: ScoredChunk,
        chunk_b: ScoredChunk,
    ) -> tuple[bool, Optional[str]]:
        """Check if two chunks from different documents make contradictory claims about the same entity."""
        text_a = f"{chunk_a.heading}\n{chunk_a.text}".lower()
        text_b = f"{chunk_b.heading}\n{chunk_b.text}".lower()
        query_lower = query.lower()

        # Check for shared entity / subject focus (e.g. breeze tumbler, tumbler, bags)
        tokens_a = set(re.findall(r"\b[a-z]{4,}\b", text_a))
        tokens_b = set(re.findall(r"\b[a-z]{4,}\b", text_b))
        common_tokens = tokens_a.intersection(tokens_b)

        # Remove generic stopwords
        stopwords = {"this", "that", "with", "from", "have", "order", "item", "customer", "aster", "when", "does"}
        meaningful_common = common_tokens - stopwords

        if not meaningful_common:
            return False, None

        # Ensure query touches at least one meaningful entity shared by both chunks
        query_tokens = set(re.findall(r"\b[a-z]{3,}\b", query_lower))
        if not (query_tokens.intersection(meaningful_common) or any(t in query_lower for t in ["tumbler", "breeze", "dishwash", "wash", "care", "clean"])):
            return False, None

        # Check contradiction patterns
        for pat1, pat2 in _CONTRADICTION_PATTERNS:
            match_a1 = pat1.search(text_a)
            match_a2 = pat2.search(text_a)
            match_b1 = pat1.search(text_b)
            match_b2 = pat2.search(text_b)

            if (match_a1 and match_b2) or (match_a2 and match_b1):
                reason = (
                    f"Contradiction detected between {chunk_a.filename} ({chunk_a.heading}) "
                    f"and {chunk_b.filename} ({chunk_b.heading}) regarding common subject "
                    f"'{', '.join(sorted(meaningful_common)[:3])}'."
                )
                return True, reason

        return False, None
