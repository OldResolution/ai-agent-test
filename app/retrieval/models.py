"""Data models for the Aster & Row RAG subsystem."""

from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field


class Document(BaseModel):
    """A loaded markdown document with parsed front-matter metadata and body."""

    document_id: str = Field(description="Unique document identifier from front-matter or derived from filename")
    filename: str = Field(description="Original markdown filename (e.g. 01-returns-policy-current.md)")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Parsed YAML front-matter metadata")
    raw_content: str = Field(description="Full raw text of the markdown file")
    body: str = Field(description="Markdown body without front matter")
    title: str = Field(default="", description="Document title extracted from front matter or first H1")


class Chunk(BaseModel):
    """An indexed chunk of text with rich structural and source metadata."""

    chunk_id: str = Field(description="Unique deterministic chunk ID")
    document_id: str = Field(description="Identifier of the source document")
    filename: str = Field(description="Original markdown filename")
    heading: str = Field(description="Most specific section heading for this passage")
    heading_path: str = Field(description="Full hierarchical path of headings (e.g. 'Returns Policy > Return Window')")
    text: str = Field(description="Actual chunk text content")
    metadata: dict[str, Any] = Field(default_factory=dict, description="Front-matter metadata preserved from source document")

    @property
    def source_reference(self) -> str:
        """Returns the formatted citation reference for this chunk."""
        if self.heading and self.heading != self.filename:
            return f"{self.filename} — {self.heading}"
        return self.filename


class ScoredChunk(BaseModel):
    """A retrieved chunk enriched with intermediate and final scores."""

    chunk: Chunk
    semantic_score: float = Field(default=0.0, description="Normalized semantic similarity score [0, 1]")
    bm25_score: float = Field(default=0.0, description="Raw or normalized BM25 lexical score")
    authority_score: float = Field(default=1.0, description="Metadata-derived authority / precedence score [0, 1]")
    final_score: float = Field(default=0.0, description="Final combined hybrid score [0, 1]")
    status: str = Field(default="active", description="Document lifecycle status (e.g. active, superseded, draft)")

    @property
    def filename(self) -> str:
        return self.chunk.filename

    @property
    def heading(self) -> str:
        return self.chunk.heading

    @property
    def heading_path(self) -> str:
        return self.chunk.heading_path

    @property
    def text(self) -> str:
        return self.chunk.text

    @property
    def metadata(self) -> dict[str, Any]:
        return self.chunk.metadata

    @property
    def source_reference(self) -> str:
        return self.chunk.source_reference


class RetrievalConfig(BaseModel):
    """Configuration for retrieval, scoring weights, and thresholds."""

    semantic_weight: float = Field(default=0.45, description="Weight for normalized semantic similarity score")
    bm25_weight: float = Field(default=0.35, description="Weight for normalized BM25 lexical score")
    authority_weight: float = Field(default=0.20, description="Weight for document authority / precedence")
    insufficient_similarity_threshold: float = Field(
        default=0.33,
        description="Minimum combined relevance required before marking evidence as insufficient"
    )
    insufficient_semantic_threshold: float = Field(
        default=0.22,
        description="Minimum semantic similarity score required for relevance"
    )
    conflict_similarity_threshold: float = Field(
        default=0.40,
        description="Minimum relevance score for considering two sources in conflict analysis"
    )
    top_k: int = Field(default=5, description="Number of top chunks to return")


class RetrievalResult(BaseModel):
    """Structured result returned by the retrieval subsystem."""

    query: str
    chunks: list[ScoredChunk] = Field(default_factory=list)
    conflict: bool = Field(default=False, description="True if genuine conflicting active authoritative sources were detected")
    conflict_sources: list[str] = Field(default_factory=list, description="Filenames of contradictory active sources")
    conflict_details: Optional[str] = Field(default=None, description="Explanation of the detected contradiction")
    insufficient: bool = Field(default=False, description="True if supplied knowledge base lacks sufficient information")
    debug_trace: Optional[str] = Field(default=None, description="Human-readable debug trace of scoring and decision process")
