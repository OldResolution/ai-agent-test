"""Markdown document loader with robust YAML front-matter parsing."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Union
import yaml

from app.retrieval.models import Document


_FRONT_MATTER_PATTERN = re.compile(r"^---\s*\r?\n(.*?)\r?\n---\s*\r?\n(.*)$", re.DOTALL)
_H1_PATTERN = re.compile(r"^#\s+(.+)$", re.MULTILINE)


class DocumentLoaderError(Exception):
    """Raised when a markdown document cannot be parsed or read."""
    pass


class MarkdownLoader:
    """Loads and parses Markdown files with optional YAML front matter."""

    def __init__(self, base_dir: Optional[Union[str, Path]] = None) -> None:
        self.base_dir = Path(base_dir) if base_dir else None

    def load_file(self, file_path: Union[str, Path]) -> Document:
        """Load and parse a single Markdown file."""
        path = Path(file_path)
        if not path.is_file():
            raise DocumentLoaderError(f"File does not exist or is not a file: {path}")

        try:
            raw_content = path.read_text(encoding="utf-8")
        except Exception as e:
            raise DocumentLoaderError(f"Failed to read file {path}: {e}") from e

        return self.parse_content(raw_content=raw_content, filename=path.name)

    def parse_content(self, raw_content: str, filename: str) -> Document:
        """Parse raw markdown content into a Document model."""
        metadata = {}
        body = raw_content

        match = _FRONT_MATTER_PATTERN.match(raw_content)
        if match:
            fm_text, body_text = match.groups()
            try:
                parsed_yaml = yaml.safe_load(fm_text)
                if isinstance(parsed_yaml, dict):
                    # Convert dates and non-string primitives to strings for clean serialization
                    metadata = self._sanitize_metadata(parsed_yaml)
                elif parsed_yaml is not None:
                    raise DocumentLoaderError(f"Front matter in {filename} is not a valid YAML mapping: {type(parsed_yaml)}")
            except yaml.YAMLError as e:
                raise DocumentLoaderError(f"Malformed YAML front matter in {filename}: {e}") from e
            body = body_text

        # Extract title from metadata or first H1 heading
        title = metadata.get("title", "")
        if not title:
            h1_match = _H1_PATTERN.search(body)
            if h1_match:
                title = h1_match.group(1).strip()
            else:
                title = Path(filename).stem.replace("-", " ").title()

        # Derive document_id if not present
        doc_id = metadata.get("document_id")
        if not doc_id:
            doc_id = Path(filename).stem

        return Document(
            document_id=str(doc_id),
            filename=filename,
            metadata=metadata,
            raw_content=raw_content,
            body=body.strip(),
            title=title,
        )

    @staticmethod
    def _sanitize_metadata(meta_dict: dict) -> dict:
        """Ensure date/time and other non-standard objects in metadata are converted to strings."""
        from datetime import date, datetime
        sanitized = {}
        for k, v in meta_dict.items():
            if isinstance(v, (date, datetime)):
                sanitized[str(k)] = v.isoformat()
            elif isinstance(v, dict):
                sanitized[str(k)] = MarkdownLoader._sanitize_metadata(v)
            else:
                sanitized[str(k)] = v
        return sanitized

    def load_directory(self, dir_path: Optional[Union[str, Path]] = None) -> list[Document]:
        """Scan a directory for .md files and parse all of them in sorted order."""
        target_dir = Path(dir_path) if dir_path else self.base_dir
        if not target_dir or not target_dir.is_dir():
            raise DocumentLoaderError(f"Directory does not exist: {target_dir}")

        md_files = sorted(target_dir.glob("*.md"))
        if not md_files:
            # Also check recursively if no top-level files
            md_files = sorted(target_dir.rglob("*.md"))

        documents = []
        for file_path in md_files:
            doc = self.load_file(file_path)
            documents.append(doc)

        return documents
