"""Tests for src/ingestion.py — PDF loading and chunking.

Tests cover:
- Loading a valid PDF and verifying page metadata
- Chunking with correct overlap and metadata preservation
- Error handling for missing, empty, and invalid files
"""

import os
import tempfile
import pytest

from src.ingestion import load_pdf, chunk_documents, ingest_pdf


class TestLoadPdf:
    """Tests for the load_pdf function."""

    def test_loads_sample_pdf(self, sample_pdf_path):
        """Test that sample.pdf loads successfully with correct metadata."""
        if not os.path.exists(sample_pdf_path):
            pytest.skip("sample.pdf not found — run scripts/generate_sample_pdf.py first")

        pages = load_pdf(sample_pdf_path)
        assert len(pages) > 0, "Should load at least one page"

        # Verify metadata
        for page in pages:
            assert "page" in page.metadata, "Each page must have page number metadata"
            assert "source" in page.metadata, "Each page must have source metadata"
            assert page.metadata["page"] >= 1, "Page numbers should be 1-indexed"
            assert page.metadata["source"] == "sample.pdf"

    def test_pages_are_one_indexed(self, sample_pdf_path):
        """Verify page numbers are 1-indexed (not 0-indexed from PyPDF)."""
        if not os.path.exists(sample_pdf_path):
            pytest.skip("sample.pdf not found")

        pages = load_pdf(sample_pdf_path)
        page_numbers = [p.metadata["page"] for p in pages]
        assert min(page_numbers) == 1, "First page should be 1, not 0"

    def test_file_not_found_raises(self):
        """Test that loading a non-existent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="PDF file not found"):
            load_pdf("/nonexistent/path/fake.pdf")

    def test_empty_file_raises(self):
        """Test that loading an empty file raises ValueError."""
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp_path = tmp.name
            # Write nothing — 0 bytes

        try:
            with pytest.raises(ValueError, match="empty"):
                load_pdf(tmp_path)
        finally:
            os.unlink(tmp_path)


class TestChunkDocuments:
    """Tests for the chunk_documents function."""

    def test_chunks_have_metadata(self, sample_pdf_path):
        """Verify that chunks preserve page and source metadata from the original pages."""
        if not os.path.exists(sample_pdf_path):
            pytest.skip("sample.pdf not found")

        pages = load_pdf(sample_pdf_path)
        chunks = chunk_documents(pages, chunk_size=500, chunk_overlap=100)

        assert len(chunks) > len(pages), "Chunking should produce more items than pages"

        for chunk in chunks:
            assert "page" in chunk.metadata
            assert "source" in chunk.metadata
            assert "chunk_index" in chunk.metadata

    def test_chunk_size_respected(self, sample_pdf_path):
        """Verify chunks don't exceed the specified size (with some tolerance for splitter)."""
        if not os.path.exists(sample_pdf_path):
            pytest.skip("sample.pdf not found")

        pages = load_pdf(sample_pdf_path)
        chunk_size = 500
        chunks = chunk_documents(pages, chunk_size=chunk_size, chunk_overlap=50)

        for chunk in chunks:
            # Allow some tolerance since RecursiveCharacterTextSplitter
            # may slightly exceed the limit at natural boundaries
            assert len(chunk.page_content) <= chunk_size * 1.1, (
                f"Chunk too large: {len(chunk.page_content)} > {chunk_size * 1.1}"
            )

    def test_empty_documents_returns_empty(self):
        """Chunking an empty list should return an empty list."""
        result = chunk_documents([])
        assert result == []

    def test_chunk_indices_are_sequential(self, sample_pdf_path):
        """Verify chunk_index metadata is sequential starting from 0."""
        if not os.path.exists(sample_pdf_path):
            pytest.skip("sample.pdf not found")

        pages = load_pdf(sample_pdf_path)
        chunks = chunk_documents(pages)
        indices = [c.metadata["chunk_index"] for c in chunks]
        assert indices == list(range(len(chunks)))


class TestIngestPdf:
    """Tests for the ingest_pdf convenience function."""

    def test_full_pipeline(self, sample_pdf_path):
        """Test the complete load → chunk pipeline."""
        if not os.path.exists(sample_pdf_path):
            pytest.skip("sample.pdf not found")

        chunks = ingest_pdf(sample_pdf_path)
        assert len(chunks) > 0
        assert all("page" in c.metadata for c in chunks)
        assert all("source" in c.metadata for c in chunks)
        assert all("chunk_index" in c.metadata for c in chunks)

    def test_custom_chunk_params(self, sample_pdf_path):
        """Test that custom chunk_size and overlap produce different results."""
        if not os.path.exists(sample_pdf_path):
            pytest.skip("sample.pdf not found")

        chunks_small = ingest_pdf(sample_pdf_path, chunk_size=300, chunk_overlap=50)
        chunks_large = ingest_pdf(sample_pdf_path, chunk_size=2000, chunk_overlap=200)

        # Smaller chunks should produce more chunks
        assert len(chunks_small) > len(chunks_large)
