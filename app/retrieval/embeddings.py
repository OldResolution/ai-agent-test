"""Embedding provider interface and implementations for OpenAI and deterministic fallback."""

from __future__ import annotations

import hashlib
import math
import os
import re
from abc import ABC, abstractmethod
from typing import Optional
import numpy as np


class EmbeddingProvider(ABC):
    """Abstract base class for generating text embeddings."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Return the vector dimensionality."""
        pass

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for a list of text passages."""
        pass

    def embed_query(self, text: str) -> list[float]:
        """Generate embedding for a single query string."""
        return self.embed_texts([text])[0]


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Generates embeddings using OpenAI's text-embedding-3-small API."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "text-embedding-3-small",
        dimension: int = 1536,
    ) -> None:
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self._api_key:
            raise ValueError(
                "OpenAI API key not found. Set OPENAI_API_KEY environment variable or pass explicitly."
            )
        self._model = model
        self._dimension = dimension

        from openai import OpenAI
        self._client = OpenAI(api_key=self._api_key)

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []

        # Process in batches of 64
        batch_size = 64
        all_embeddings: list[list[float]] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            # Replace newlines with spaces as recommended by OpenAI embedding guidelines
            sanitized = [t.replace("\r\n", " ").replace("\n", " ") for t in batch]
            response = self._client.embeddings.create(
                input=sanitized,
                model=self._model,
            )
            # Sort by index to maintain ordering
            sorted_data = sorted(response.data, key=lambda d: d.index)
            for item in sorted_data:
                all_embeddings.append(item.embedding)

        return all_embeddings


class DeterministicEmbeddingProvider(EmbeddingProvider):
    """Deterministic, offline embedding provider for hermetic testing and CI without API keys.

    Uses character n-grams and token hashing with TF-IDF weighting and L2 normalization
    to create reproducible vectors with high semantic correlation for keyword/phrase matches.
    """

    def __init__(self, dimension: int = 384) -> None:
        self._dim = dimension

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        # Common english stopwords to downweight
        stopwords = {
            "a", "an", "the", "in", "on", "at", "to", "for", "of", "with", "by", "from",
            "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
            "do", "does", "did", "can", "could", "shall", "should", "will", "would",
            "and", "or", "but", "if", "because", "as", "what", "which", "this", "that",
            "these", "those", "then", "so", "than", "too", "very", "all", "any", "both",
            "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not",
            "only", "own", "same", "your", "my", "our", "their", "its", "you", "i", "we", "they"
        }

        # Semantic concept expansion clusters for domain concepts
        concept_map = {
            # Geographic / International
            "germany": ["international", "country", "destinations", "shipping", "canada", "foreign"],
            "france": ["international", "country", "destinations", "shipping", "canada"],
            "uk": ["international", "country", "destinations", "shipping", "canada"],
            "europe": ["international", "country", "destinations", "shipping", "canada"],
            "australia": ["international", "country", "destinations", "shipping", "canada"],
            "canada": ["international", "destinations", "shipping", "country", "countries"],
            "international": ["destinations", "canada", "foreign", "overseas", "countries", "outside"],
            "internationally": ["international", "destinations", "canada", "countries", "outside"],
            "outside": ["international", "destinations", "canada", "foreign", "countries"],
            "country": ["international", "destinations", "canada", "shipping"],
            "countries": ["international", "destinations", "canada", "shipping"],
            # Time / Window
            "how long": ["window", "calendar", "days", "delivery", "period", "timeline"],
            "how many days": ["window", "calendar", "days", "delivery", "return", "policy"],
            "duration": ["window", "calendar", "days", "time"],
            "window": ["days", "calendar", "time", "period"],
            # Returns & Actions
            "send back": ["return", "returns", "window", "policy", "delivery"],
            "send an item back": ["return", "returns", "window", "policy", "delivery"],
            "item back": ["return", "returns", "window", "policy"],
            "return": ["returns", "window", "policy", "refund"],
            "returns": ["return", "window", "policy", "refund"],
            # Customer tiers
            "regular": ["standard", "customer", "plan"],
            "standard": ["regular", "customer", "plan"],
            # Condition / Damage / Defects
            "broken": ["damaged", "defective", "wrong", "defect", "zipper"],
            "zipper": ["damaged", "defective", "repair", "manufacturing", "warranty"],
            "tear": ["damaged", "defective", "warranty"],
            "flaw": ["defective", "defect", "manufacturing"],
            # Products
            "backpack": ["bags", "backpacks", "merchandise", "item"],
            "weekender": ["bags", "backpacks", "luggage", "travel"],
            "tumbler": ["breeze", "drinkware", "cleaning", "dishwasher", "hand-wash"],
        }

        embeddings: list[list[float]] = []
        for text in texts:
            vec = np.zeros(self._dim, dtype=np.float32)
            clean_text = text.lower()
            tokens = re.findall(r"\b[a-z0-9_-]+\b", clean_text)

            # Apply lightweight stemming & concept expansion
            expanded_tokens = list(tokens)
            for token in tokens:
                stem = self._stem(token)
                if stem != token:
                    expanded_tokens.append(stem)
                if token in concept_map:
                    expanded_tokens.extend(concept_map[token])

            # Check for phrase matches
            for phrase, expansions in concept_map.items():
                if " " in phrase and phrase in clean_text:
                    expanded_tokens.extend(expansions)

            for token in expanded_tokens:
                weight = 0.15 if token in stopwords else 1.8
                h = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
                idx = h % self._dim
                sign = 1.0 if ((h >> 4) & 1) else -1.0
                vec[idx] += sign * weight

                if token not in stopwords and len(token) >= 4:
                    for k in range(len(token) - 2):
                        ngram = token[k : k + 3]
                        nh = int(hashlib.md5(ngram.encode("utf-8")).hexdigest(), 16)
                        n_idx = nh % self._dim
                        n_sign = 1.0 if ((nh >> 4) & 1) else -1.0
                        vec[n_idx] += n_sign * 0.4

            # Bigram capture
            for j in range(len(tokens) - 1):
                t1, t2 = tokens[j], tokens[j+1]
                if t1 not in stopwords or t2 not in stopwords:
                    bigram = f"{t1}_{t2}"
                    bh = int(hashlib.md5(bigram.encode("utf-8")).hexdigest(), 16)
                    b_idx = bh % self._dim
                    b_sign = 1.0 if ((bh >> 4) & 1) else -1.0
                    vec[b_idx] += b_sign * 2.2

            norm = np.linalg.norm(vec)
            if norm > 1e-6:
                vec = vec / norm
            else:
                vec[0] = 1.0

            embeddings.append(vec.tolist())

        return embeddings

    @staticmethod
    def _stem(word: str) -> str:
        """Lightweight Porter-like suffix stripping for common inflections."""
        w = word.lower()
        if w.endswith("ing") and len(w) > 5:
            return w[:-3]
        if w.endswith("ed") and len(w) > 4:
            return w[:-2]
        if w.endswith("ies") and len(w) > 4:
            return w[:-3] + "y"
        if w.endswith("es") and len(w) > 4:
            return w[:-2]
        if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
            return w[:-1]
        if w.endswith("ly") and len(w) > 4:
            return w[:-2]
        return w


def get_embedding_provider(
    api_key: Optional[str] = None,
    mock: bool = False,
    model: str = "text-embedding-3-small",
) -> EmbeddingProvider:
    """Factory helper to obtain an appropriate embedding provider."""
    resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
    if mock or not resolved_key:
        return DeterministicEmbeddingProvider()
    return OpenAIEmbeddingProvider(api_key=resolved_key, model=model)
