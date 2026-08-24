"""Tests for detecting genuine active source conflicts."""

from app.retrieval.retriever import HybridRetriever


def test_breeze_tumbler_dishwasher_conflict_detection(retriever: HybridRetriever):
    """Test that the direct conflict between 11-product-care and 12-breeze-tumbler-product-card is detected."""
    query = "Can I put the entire Breeze Tumbler in the dishwasher?"
    result = retriever.search(query, top_k=5)

    assert not result.insufficient
    assert result.conflict is True, "Expected conflict=True for Breeze Tumbler cleaning query"
    assert "11-product-care.md" in result.conflict_sources
    assert "12-breeze-tumbler-product-card.md" in result.conflict_sources

    # Ensure both conflicting documents are present in retrieved chunks
    retrieved_files = {c.filename for c in result.chunks}
    assert "11-product-care.md" in retrieved_files
    assert "12-breeze-tumbler-product-card.md" in retrieved_files


def test_no_false_conflict_on_standard_queries(retriever: HybridRetriever):
    """Verify that standard non-conflicting queries do not trigger false conflict flags."""
    non_conflicting = [
        "What is the standard return window?",
        "Do you ship to Canada?",
        "How do I clean my fabric backpack?",
        "What is the warranty period for bags?",
    ]

    for q in non_conflicting:
        res = retriever.search(q, top_k=5)
        assert res.conflict is False, f"Unexpected conflict flagged for non-conflicting query: '{q}'"
        assert res.conflict_sources == []
