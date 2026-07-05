"""PDF ingestion module — loads, chunks, and prepares documents for the vector store.

This module handles the entire document ingestion pipeline:
1. Load PDF files using PyPDF
2. Split into overlapping chunks to preserve context at chunk boundaries
3. Attach page number metadata to each chunk for source citation

Design decisions:
- chunk_size=1000 and overlap=200 balance retrieval precision with context completeness
- Page numbers are preserved as metadata so RAG answers can cite specific pages
- All file I/O is wrapped in try/except for graceful error handling
"""

import logging
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

logger = logging.getLogger(__name__)

# --- Configuration ---
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 200


def load_pdf(file_path: str) -> list[Document]:
    """Load a PDF file and return a list of Document objects, one per page.

    Each Document's metadata includes the source filename and page number
    (0-indexed from PyPDF, converted to 1-indexed for user display).

    Args:
        file_path: Absolute or relative path to the PDF file.

    Returns:
        List of Document objects with page content and metadata.

    Raises:
        FileNotFoundError: If the PDF file doesn't exist at the given path.
        ValueError: If the PDF is empty or contains no extractable text.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {file_path}. "
            "Please check the file path and try again."
        )

    if path.stat().st_size == 0:
        raise ValueError(
            f"PDF file is empty (0 bytes): {file_path}. "
            "Please upload a valid PDF document."
        )

    try:
        loader = PyPDFLoader(str(path))
        pages = loader.load()
    except Exception as e:
        logger.error(f"Failed to load PDF '{file_path}': {e}")
        raise ValueError(
            f"Could not read PDF file: {file_path}. "
            f"The file may be corrupted or password-protected. Error: {str(e)}"
        ) from e

    if not pages:
        raise ValueError(
            f"PDF file contains no extractable text: {file_path}. "
            "The file may be image-only (scanned). "
            "Please upload a PDF with selectable text."
        )

    # Convert PyPDF's 0-indexed page numbers to 1-indexed for user display
    for doc in pages:
        doc.metadata["page"] = doc.metadata.get("page", 0) + 1
        doc.metadata["source"] = path.name

    logger.info(f"Loaded {len(pages)} pages from '{path.name}'")
    return pages


def chunk_documents(
    documents: list[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """Split documents into overlapping chunks while preserving metadata.

    Uses RecursiveCharacterTextSplitter which tries to split on natural
    boundaries (paragraphs, sentences, words) before falling back to
    character-level splits. This produces more coherent chunks than
    a naive character split.

    Args:
        documents: List of Document objects (typically one per PDF page).
        chunk_size: Maximum number of characters per chunk.
        chunk_overlap: Number of overlapping characters between consecutive chunks.
            Overlap ensures that information at chunk boundaries isn't lost.

    Returns:
        List of Document objects, each representing one chunk with preserved
        metadata (source filename, page number).
    """
    if not documents:
        logger.warning("No documents provided for chunking.")
        return []

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        # Split hierarchy: try paragraphs first, then sentences, then words
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = text_splitter.split_documents(documents)

    # Add chunk index to metadata for traceability
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    logger.info(
        f"Split {len(documents)} documents into {len(chunks)} chunks "
        f"(chunk_size={chunk_size}, overlap={chunk_overlap})"
    )
    return chunks


def ingest_pdf(
    file_path: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[Document]:
    """Full ingestion pipeline: load PDF → chunk → return documents ready for embedding.

    This is the main entry point for the ingestion module. It combines
    loading and chunking into a single call for convenience.

    Args:
        file_path: Path to the PDF file to ingest.
        chunk_size: Maximum characters per chunk (default: 1000).
        chunk_overlap: Overlap between consecutive chunks (default: 200).

    Returns:
        List of chunked Document objects with metadata including:
        - source: Original filename
        - page: 1-indexed page number
        - chunk_index: Sequential chunk number

    Example:
        >>> chunks = ingest_pdf("data/sample.pdf")
        >>> print(f"Created {len(chunks)} chunks")
        >>> print(f"First chunk from page {chunks[0].metadata['page']}")
    """
    pages = load_pdf(file_path)
    chunks = chunk_documents(pages, chunk_size, chunk_overlap)
    return chunks


# --- CLI entry point for standalone testing ---
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    pdf_path = sys.argv[1] if len(sys.argv) > 1 else "data/sample.pdf"
    try:
        chunks = ingest_pdf(pdf_path)
        print(f"\n✅ Successfully ingested '{pdf_path}'")
        print(f"   Pages loaded: {max(c.metadata['page'] for c in chunks)}")
        print(f"   Chunks created: {len(chunks)}")
        print("\n   Sample chunk (index 0):")
        print(f"   Page: {chunks[0].metadata['page']}")
        print(f"   Text: {chunks[0].page_content[:200]}...")
    except (FileNotFoundError, ValueError) as e:
        print(f"\n❌ Error: {e}")
        sys.exit(1)
