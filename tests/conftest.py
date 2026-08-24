"""Shared pytest fixtures for the Aster & Row RAG test suite."""

import pytest
from pathlib import Path
from app.retrieval.chunker import MarkdownChunker
from app.retrieval.embeddings import DeterministicEmbeddingProvider
from app.retrieval.index import KnowledgeIndex
from app.retrieval.loader import MarkdownLoader
from app.retrieval.models import RetrievalConfig
from app.retrieval.retriever import HybridRetriever


@pytest.fixture(scope="session")
def project_root() -> Path:
    return Path(__file__).resolve().parent.parent


@pytest.fixture(scope="session")
def kb_dir(project_root: Path) -> Path:
    return project_root / "knowledge-base"


@pytest.fixture(scope="session")
def storage_dir(project_root: Path) -> Path:
    return project_root / "storage"


@pytest.fixture(scope="session")
def embedding_provider() -> DeterministicEmbeddingProvider:
    return DeterministicEmbeddingProvider(dimension=384)


@pytest.fixture(scope="session")
def loaded_documents(kb_dir: Path) -> list:
    loader = MarkdownLoader(base_dir=kb_dir)
    return loader.load_directory()


@pytest.fixture(scope="session")
def all_chunks(loaded_documents: list) -> list:
    chunker = MarkdownChunker()
    return chunker.chunk_documents(loaded_documents)


@pytest.fixture(scope="session")
def knowledge_index(all_chunks: list, embedding_provider: DeterministicEmbeddingProvider) -> KnowledgeIndex:
    return KnowledgeIndex.build(chunks=all_chunks, embedding_provider=embedding_provider)


@pytest.fixture(scope="session")
def retriever(knowledge_index: KnowledgeIndex) -> HybridRetriever:
    config = RetrievalConfig()
    return HybridRetriever(index=knowledge_index, config=config)
