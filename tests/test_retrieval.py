"""Tests for hybrid retrieval across query variations and debug representation."""

from app.retrieval.retriever import HybridRetriever


def test_return_window_query_variations(retriever: HybridRetriever):
    """Test that natural language variations for returns retrieve current return policy."""
    queries = [
        "What is the return window?",
        "How long do I have to return something?",
        "How many days do I have to send an item back?",
        "What is the standard return period for regular customers?",
    ]

    for q in queries:
        result = retriever.search(q, top_k=5)
        assert not result.insufficient, f"Query '{q}' was marked insufficient"
        assert len(result.chunks) > 0

        # Current returns policy should be among the top results
        retrieved_files = [c.filename for c in result.chunks[:3]]
        assert "01-returns-policy-current.md" in retrieved_files, (
            f"Expected '01-returns-policy-current.md' in top results for query: '{q}', got {retrieved_files}"
        )


def test_international_shipping_query_variations(retriever: HybridRetriever):
    """Test international shipping query variations."""
    queries = [
        "Do you ship internationally?",
        "Can you ship orders outside the country?",
        "What countries do you ship to?",
        "Do you ship to Canada?",
    ]

    for q in queries:
        result = retriever.search(q, top_k=5)
        assert not result.insufficient, f"Query '{q}' was marked insufficient"

        retrieved_files = [c.filename for c in result.chunks[:3]]
        assert "06-international-shipping.md" in retrieved_files, (
            f"Expected '06-international-shipping.md' in top results for query: '{q}', got {retrieved_files}"
        )


def test_product_and_warranty_queries(retriever: HybridRetriever):
    """Test product care and warranty retrieval."""
    # Warranty query
    res_warranty = retriever.search("Do all products have a lifetime warranty?", top_k=3)
    assert not res_warranty.insufficient
    assert any(c.filename == "07-warranty.md" for c in res_warranty.chunks[:2])

    # Product care query
    res_care = retriever.search("How do I clean my bags and backpacks?", top_k=3)
    assert not res_care.insufficient
    assert any(c.filename == "11-product-care.md" for c in res_care.chunks[:2])


def test_retrieval_debug_representation(retriever: HybridRetriever):
    """Test that debug trace contains all required observability fields."""
    res = retriever.search("What is the return window?", top_k=3)
    trace = res.debug_trace
    assert trace is not None

    # Check key debug fields
    assert 'Query: "What is the return window?"' in trace
    assert "Heading:" in trace
    assert "Heading Path:" in trace
    assert "Semantic:" in trace
    assert "BM25:" in trace
    assert "Authority:" in trace
    assert "Final:" in trace
    assert "Status:" in trace
    assert "Source Citation: Source:" in trace
