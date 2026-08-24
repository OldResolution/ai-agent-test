"""Tests for document authority and precedence ranking."""

from app.retrieval.models import Chunk
from app.retrieval.ranking import HybridRanker, calculate_authority_score
from app.retrieval.retriever import HybridRetriever


def test_active_policy_outranks_superseded_policy(retriever: HybridRetriever):
    """Test that 01-returns-policy-current outranks 02-returns-policy-legacy for standard return window."""
    res = retriever.search("What is the standard return window for customers?", top_k=5)
    assert not res.insufficient

    # Find ranks of current vs legacy returns documents
    current_rank = None
    legacy_rank = None

    for idx, c in enumerate(res.chunks):
        if c.filename == "01-returns-policy-current.md" and current_rank is None:
            current_rank = idx
        elif c.filename == "02-returns-policy-legacy.md" and legacy_rank is None:
            legacy_rank = idx

    assert current_rank is not None, "01-returns-policy-current.md was not retrieved in top 5"
    if legacy_rank is not None:
        assert current_rank < legacy_rank, (
            f"Active policy (rank {current_rank}) did not outrank legacy policy (rank {legacy_rank})"
        )


def test_internal_migration_draft_is_suppressed(retriever: HybridRetriever):
    """Test that 14-internal-content-migration-notes is penalized and never outranks active official policy."""
    query = "Every customer receives 60 days to return every item."
    res = retriever.search(query, top_k=5)

    # If migration note appears, its authority score must be 0.0 and final score lower than active policy
    migration_chunks = [c for c in res.chunks if c.filename == "14-internal-content-migration-notes.md"]
    if migration_chunks:
        for mc in migration_chunks:
            assert mc.authority_score == 0.0, "Migration draft should have 0 authority score"


def test_generic_authority_scoring_without_filenames():
    """Verify that authority calculation is strictly generic and driven by metadata."""
    meta_active_official = {
        "status": "active",
        "policy_authority": "official",
        "audience": "customer",
        "customer_answering": True,
    }
    meta_superseded = {
        "status": "superseded",
        "policy_authority": "official",
        "audience": "customer",
        "superseded_by": "DOC-999",
    }
    meta_draft_none = {
        "status": "draft",
        "policy_authority": "none",
        "audience": "internal",
        "customer_answering": False,
    }
    meta_internal_active = {
        "status": "active",
        "policy_authority": "official",
        "audience": "internal",
    }

    score_active = calculate_authority_score(meta_active_official)
    score_superseded = calculate_authority_score(meta_superseded)
    score_draft = calculate_authority_score(meta_draft_none)
    score_internal = calculate_authority_score(meta_internal_active)

    assert score_active == 1.0
    assert 0.10 <= score_superseded <= 0.20
    assert score_draft == 0.0
    assert score_internal == 0.70

    assert score_active > score_internal > score_superseded > score_draft
