"""Tests for the MarkdownLoader."""

import hashlib
from pathlib import Path
import pytest
from app.retrieval.loader import DocumentLoaderError, MarkdownLoader


def test_markdown_loader_discovers_all_files(kb_dir: Path):
    """Test that all 14 knowledge base markdown files are discovered and loaded."""
    loader = MarkdownLoader(base_dir=kb_dir)
    docs = loader.load_directory()
    assert len(docs) == 14
    filenames = {d.filename for d in docs}
    expected_files = {
        "01-returns-policy-current.md",
        "02-returns-policy-legacy.md",
        "03-final-sale-and-promotions.md",
        "04-damaged-or-wrong-items.md",
        "05-domestic-shipping.md",
        "06-international-shipping.md",
        "07-warranty.md",
        "08-order-changes-and-cancellations.md",
        "09-trailplus-membership.md",
        "10-gift-cards-and-price-adjustments.md",
        "11-product-care.md",
        "12-breeze-tumbler-product-card.md",
        "13-support-escalation.md",
        "14-internal-content-migration-notes.md",
    }
    assert filenames == expected_files


def test_front_matter_parsing_and_metadata_preservation(kb_dir: Path):
    """Test that YAML front matter is parsed accurately and all fields preserved."""
    loader = MarkdownLoader(base_dir=kb_dir)
    doc = loader.load_file(kb_dir / "01-returns-policy-current.md")

    assert doc.document_id == "RET-2026-01"
    assert doc.title == "Returns Policy"
    assert doc.metadata["status"] == "active"
    assert doc.metadata["policy_authority"] == "official"
    assert doc.metadata["audience"] == "customer"
    assert doc.metadata["effective_date"] == "2026-04-01"
    assert doc.metadata["supersedes"] == "RET-2024-01"
    assert "Standard return window" in doc.body


def test_document_without_front_matter():
    """Test that markdown files without front matter parse cleanly."""
    loader = MarkdownLoader()
    raw = "# Simple Guide\n\nThis is a simple guide without front matter."
    doc = loader.parse_content(raw_content=raw, filename="guide.md")

    assert doc.filename == "guide.md"
    assert doc.title == "Simple Guide"
    assert doc.metadata == {}
    assert "This is a simple guide" in doc.body


def test_malformed_yaml_raises_error():
    """Test that malformed YAML in front matter raises DocumentLoaderError rather than failing silently."""
    loader = MarkdownLoader()
    bad_raw = "---\ntitle: [unclosed list\nstatus: active\n---\n# Content"
    with pytest.raises(DocumentLoaderError) as exc_info:
        loader.parse_content(raw_content=bad_raw, filename="bad.md")
    assert "Malformed YAML front matter" in str(exc_info.value)


def test_source_files_remain_unmodified(kb_dir: Path):
    """Verify that source markdown files are never mutated during loading."""
    # Compute initial hashes
    hashes_before = {}
    for f in kb_dir.glob("*.md"):
        hashes_before[f.name] = hashlib.sha256(f.read_bytes()).hexdigest()

    # Run loader
    loader = MarkdownLoader(base_dir=kb_dir)
    _ = loader.load_directory()

    # Re-verify hashes
    for f in kb_dir.glob("*.md"):
        hash_after = hashlib.sha256(f.read_bytes()).hexdigest()
        assert hashes_before[f.name] == hash_after, f"Source file {f.name} was modified!"
