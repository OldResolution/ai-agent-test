"""Heading-aware Markdown chunker that preserves document hierarchy and context."""

from __future__ import annotations

import re
from typing import Optional
from app.retrieval.models import Chunk, Document


_HEADING_REGEX = re.compile(r"^(#{1,6})\s+(.+)$")
_SENTENCE_SPLIT_REGEX = re.compile(r"(?<=[.!?])\s+")


class MarkdownChunker:
    """Chunks markdown documents based on heading hierarchy and section boundaries."""

    def __init__(
        self,
        max_chunk_chars: int = 1000,
        min_chunk_chars: int = 40,
        overlap_sentences: int = 1,
    ) -> None:
        self.max_chunk_chars = max_chunk_chars
        self.min_chunk_chars = min_chunk_chars
        self.overlap_sentences = overlap_sentences

    def chunk_document(self, document: Document) -> list[Chunk]:
        """Split a document into heading-aware semantic chunks."""
        sections = self._extract_sections(document)
        chunks: list[Chunk] = []

        for section_idx, sec in enumerate(sections):
            heading = sec["heading"]
            heading_path = sec["heading_path"]
            raw_text = sec["text"].strip()

            if not raw_text:
                continue

            # If section fits within max_chunk_chars, emit as single chunk
            if len(raw_text) <= self.max_chunk_chars:
                slug = self._slugify(heading)
                chunk_id = f"{document.document_id}#{slug}#{section_idx}"
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        document_id=document.document_id,
                        filename=document.filename,
                        heading=heading,
                        heading_path=heading_path,
                        text=raw_text,
                        metadata=dict(document.metadata),
                    )
                )
            else:
                # Sub-chunk the section into smaller paragraphs/sentences
                sub_chunks = self._subchunk_section(
                    raw_text=raw_text,
                    document=document,
                    heading=heading,
                    heading_path=heading_path,
                    base_section_idx=section_idx,
                )
                chunks.extend(sub_chunks)

        return chunks

    def chunk_documents(self, documents: list[Document]) -> list[Chunk]:
        """Chunk multiple documents."""
        all_chunks: list[Chunk] = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks

    def _extract_sections(self, document: Document) -> list[dict[str, str]]:
        """Parse markdown body into sections based on headings."""
        lines = document.body.splitlines()
        sections: list[dict[str, str]] = []

        heading_stack: list[tuple[int, str]] = []
        # If document has a title, seed stack or use as default
        default_heading = document.title or document.filename

        current_lines: list[str] = []
        current_heading = default_heading
        current_heading_path = default_heading

        for line in lines:
            match = _HEADING_REGEX.match(line.strip())
            if match:
                # Flush previous section
                if current_lines:
                    text = "\n".join(current_lines).strip()
                    if text:
                        sections.append({
                            "heading": current_heading,
                            "heading_path": current_heading_path,
                            "text": text,
                        })
                    current_lines = []

                level = len(match.group(1))
                heading_text = match.group(2).strip()

                # Pop headings at or below the new level
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()

                heading_stack.append((level, heading_text))

                current_heading = heading_text
                current_heading_path = " > ".join(h[1] for h in heading_stack)
            else:
                current_lines.append(line)

        # Flush final section
        if current_lines:
            text = "\n".join(current_lines).strip()
            if text:
                sections.append({
                    "heading": current_heading,
                    "heading_path": current_heading_path,
                    "text": text,
                })

        return sections

    def _subchunk_section(
        self,
        raw_text: str,
        document: Document,
        heading: str,
        heading_path: str,
        base_section_idx: int,
    ) -> list[Chunk]:
        """Split a large section text along paragraphs and sentence boundaries."""
        paragraphs = [p.strip() for p in raw_text.split("\n\n") if p.strip()]
        chunks: list[Chunk] = []
        slug = self._slugify(heading)
        sub_idx = 0

        current_piece = ""

        def flush_piece(piece: str) -> None:
            nonlocal sub_idx
            p_clean = piece.strip()
            if p_clean:
                chunk_id = f"{document.document_id}#{slug}#{base_section_idx}_{sub_idx}"
                chunks.append(
                    Chunk(
                        chunk_id=chunk_id,
                        document_id=document.document_id,
                        filename=document.filename,
                        heading=heading,
                        heading_path=heading_path,
                        text=p_clean,
                        metadata=dict(document.metadata),
                    )
                )
                sub_idx += 1

        for p in paragraphs:
            if len(p) <= self.max_chunk_chars:
                if not current_piece:
                    current_piece = p
                elif len(current_piece) + len(p) + 2 <= self.max_chunk_chars:
                    current_piece += "\n\n" + p
                else:
                    flush_piece(current_piece)
                    current_piece = p
            else:
                # Paragraph itself is too large, flush current and split by sentences
                if current_piece:
                    flush_piece(current_piece)
                    current_piece = ""

                sentences = _SENTENCE_SPLIT_REGEX.split(p)
                for s in sentences:
                    s_clean = s.strip()
                    if not s_clean:
                        continue
                    if not current_piece:
                        current_piece = s_clean
                    elif len(current_piece) + len(s_clean) + 1 <= self.max_chunk_chars:
                        current_piece += " " + s_clean
                    else:
                        flush_piece(current_piece)
                        # If single sentence still exceeds max_chunk_chars, chunk by characters
                        if len(s_clean) > self.max_chunk_chars:
                            for k in range(0, len(s_clean), self.max_chunk_chars):
                                flush_piece(s_clean[k : k + self.max_chunk_chars])
                            current_piece = ""
                        else:
                            current_piece = s_clean

        if current_piece:
            flush_piece(current_piece)

        return chunks

    @staticmethod
    def _slugify(text: str) -> str:
        """Create a clean alphanumeric slug from a heading string."""
        s = text.lower()
        s = re.sub(r"[^\w\s-]", "", s)
        return re.sub(r"[-\s]+", "-", s).strip("-")[:32]
