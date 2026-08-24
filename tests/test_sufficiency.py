"""Tests for detecting insufficient information and out-of-domain queries."""

from app.retrieval.retriever import HybridRetriever


def test_insufficient_evidence_for_out_of_domain_queries(retriever: HybridRetriever):
    """Test that out-of-domain queries without adequate knowledge base coverage return insufficient=True."""
    insufficient_queries = [
        "Are all fabrics and adhesives in your bags vegan?",
        "What is the weather forecast in Tokyo tomorrow?",
        "Do you sell prescription eyeglasses?",
        "Can I pay using Bitcoin or Ethereum cryptocurrency?",
    ]

    for q in insufficient_queries:
        res = retriever.search(q, top_k=5)
        assert res.insufficient is True, f"Expected insufficient=True for query '{q}', got insufficient={res.insufficient}"


def test_sufficient_evidence_for_in_domain_queries(retriever: HybridRetriever):
    """Test that valid knowledge base queries return insufficient=False."""
    in_domain_queries = [
        "What is the return window?",
        "Do you ship internationally?",
        "What is covered under the warranty?",
        "Can I cancel an order that is processing?",
        "How do TrailPlus member benefits work for returns?",
    ]

    for q in in_domain_queries:
        res = retriever.search(q, top_k=5)
        assert res.insufficient is False, f"Expected insufficient=False for query '{q}', got insufficient={res.insufficient}"


def test_empty_query_returns_insufficient(retriever: HybridRetriever):
    """Test that an empty or whitespace query safely returns insufficient=True with no chunks."""
    res = retriever.search("   ")
    assert res.insufficient is True
    assert res.chunks == []
