"""Tests for the MarkdownChunker."""

from app.retrieval.chunker import MarkdownChunker
from app.retrieval.models import Document


def test_heading_hierarchy_preservation():
    """Test that nested headings preserve full heading_path hierarchy."""
    doc = Document(
        document_id="TEST-01",
        filename="test-policy.md",
        metadata={"status": "active", "policy_authority": "official"},
        raw_content="",
        body="""# Returns Policy

## Return Window

Standard returns must be initiated within 30 days.

## Exceptions

### Final Sale Items

Final sale items cannot be returned.

### Gift Cards

Gift cards are non-refundable.
""",
        title="Returns Policy",
    )

    chunker = MarkdownChunker()
    chunks = chunker.chunk_document(doc)

    assert len(chunks) == 3

    # First section
    assert chunks[0].heading == "Return Window"
    assert chunks[0].heading_path == "Returns Policy > Return Window"
    assert "within 30 days" in chunks[0].text
    assert chunks[0].filename == "test-policy.md"
    assert chunks[0].metadata["status"] == "active"

    # Nested section 1
    assert chunks[1].heading == "Final Sale Items"
    assert chunks[1].heading_path == "Returns Policy > Exceptions > Final Sale Items"
    assert "Final sale items" in chunks[1].text

    # Nested section 2
    assert chunks[2].heading == "Gift Cards"
    assert chunks[2].heading_path == "Returns Policy > Exceptions > Gift Cards"
    assert "Gift cards are non-refundable" in chunks[2].text


def test_large_section_subchunking_preserves_metadata():
    """Test that sections exceeding max_chunk_chars are cleanly sub-chunked."""
    long_paragraph_1 = "This is sentence one of paragraph one. " * 15
    long_paragraph_2 = "This is sentence one of paragraph two. " * 15

    doc = Document(
        document_id="LONG-01",
        filename="long-doc.md",
        metadata={"category": "test"},
        raw_content="",
        body=f"""# Comprehensive Guide

## Section One

{long_paragraph_1}

{long_paragraph_2}
""",
        title="Comprehensive Guide",
    )

    chunker = MarkdownChunker(max_chunk_chars=300)
    chunks = chunker.chunk_document(doc)

    assert len(chunks) > 1
    for c in chunks:
        assert c.filename == "long-doc.md"
        assert c.heading == "Section One"
        assert c.heading_path == "Comprehensive Guide > Section One"
        assert c.metadata["category"] == "test"
        assert len(c.text) <= 350


def test_chunk_source_reference_property():
    """Test chunk.source_reference property format."""
    doc = Document(
        document_id="RET-01",
        filename="01-returns-policy.md",
        metadata={},
        raw_content="",
        body="""# Returns Policy

## Standard return window

Customers may return within 30 days.
""",
        title="Returns Policy",
    )

    chunker = MarkdownChunker()
    chunks = chunker.chunk_document(doc)
    assert len(chunks) == 1
    assert chunks[0].source_reference == "01-returns-policy.md — Standard return window"
