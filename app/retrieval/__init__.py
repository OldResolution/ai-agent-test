"""RAG Subsystem for Aster & Row Support Agent."""

from app.retrieval.chunker import MarkdownChunker
from app.retrieval.conflicts import ConflictDetector
from app.retrieval.embeddings import (
    DeterministicEmbeddingProvider,
    EmbeddingProvider,
    OpenAIEmbeddingProvider,
    get_embedding_provider,
)
from app.retrieval.index import KnowledgeIndex
from app.retrieval.loader import DocumentLoaderError, MarkdownLoader
from app.retrieval.models import (
    Chunk,
    Document,
    RetrievalConfig,
    RetrievalResult,
    ScoredChunk,
)
from app.retrieval.ranking import HybridRanker, calculate_authority_score
from app.retrieval.retriever import HybridRetriever

__all__ = [
    "Chunk",
    "ConflictDetector",
    "DeterministicEmbeddingProvider",
    "Document",
    "DocumentLoaderError",
    "EmbeddingProvider",
    "HybridRanker",
    "HybridRetriever",
    "KnowledgeIndex",
    "MarkdownChunker",
    "MarkdownLoader",
    "OpenAIEmbeddingProvider",
    "RetrievalConfig",
    "RetrievalResult",
    "ScoredChunk",
    "calculate_authority_score",
    "get_embedding_provider",
]
