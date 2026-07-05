"""Tests for src/vectorstore.py — vector store operations and RAG pipeline.

Tests cover:
- Embedding function creation and configuration
- Document storage and retrieval
- RAG query pipeline (retrieve → prompt → generate → cite)
- Error handling for missing API keys and failed operations

All Gemini API calls and ChromaDB operations are mocked — no network or quota burn.
"""

from unittest.mock import patch, MagicMock

from langchain_core.documents import Document

from src.vectorstore import (
    _get_api_key,
    get_embedding_function,
    add_documents,
    query_documents,
    rag_query,
    ingest_and_store,
)


class TestGetApiKey:
    """Tests for the _get_api_key helper."""

    @patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"})
    def test_returns_key_when_set(self):
        """Should return the API key from environment."""
        assert _get_api_key() == "test-key"

    @patch.dict("os.environ", {}, clear=True)
    def test_raises_when_missing(self):
        """Should raise EnvironmentError when key is missing."""
        try:
            _get_api_key()
            assert False, "Expected EnvironmentError"
        except EnvironmentError as e:
            assert "GOOGLE_API_KEY" in str(e)


class TestGetEmbeddingFunction:
    """Tests for the get_embedding_function factory."""

    @patch("src.vectorstore.GoogleGenerativeAIEmbeddings")
    @patch.dict("os.environ", {
        "GOOGLE_API_KEY": "test-key",
        "GEMINI_EMBEDDING_MODEL": "gemini-embedding-2",
    })
    def test_uses_configured_model(self, mock_embeddings_cls):
        """Should create embeddings with the configured model name."""
        get_embedding_function()
        mock_embeddings_cls.assert_called_once()
        call_kwargs = mock_embeddings_cls.call_args
        assert "models/gemini-embedding-2" in str(call_kwargs)

    @patch("src.vectorstore.GoogleGenerativeAIEmbeddings")
    @patch.dict("os.environ", {"GOOGLE_API_KEY": "test-key"}, clear=True)
    def test_uses_default_model_when_not_set(self, mock_embeddings_cls):
        """Should fall back to gemini-embedding-2 when env var not set."""
        get_embedding_function()
        mock_embeddings_cls.assert_called_once()
        call_kwargs = mock_embeddings_cls.call_args
        assert "models/gemini-embedding-2" in str(call_kwargs)


class TestAddDocuments:
    """Tests for the add_documents function."""

    def test_empty_documents_raises(self):
        """Should raise ValueError when given empty list."""
        try:
            add_documents([])
            assert False, "Expected ValueError"
        except ValueError as e:
            assert "No documents" in str(e)

    @patch("src.vectorstore.get_embedding_function")
    @patch("src.vectorstore.Chroma.from_documents")
    def test_adds_documents_successfully(self, mock_from_docs, mock_embed_fn):
        """Should call Chroma.from_documents with the provided documents."""
        docs = [Document(page_content="Test content", metadata={"page": 1})]
        mock_from_docs.return_value = MagicMock()

        result = add_documents(docs, persist_directory="/tmp/test_chroma")
        mock_from_docs.assert_called_once()
        assert result is not None


class TestQueryDocuments:
    """Tests for the query_documents function."""

    @patch("src.vectorstore.get_vectorstore")
    def test_returns_formatted_results(self, mock_get_vs):
        """Should return a list of dicts with text, metadata, and score."""
        mock_vs = MagicMock()
        mock_doc = Document(
            page_content="Solar panels are efficient.",
            metadata={"source": "test.pdf", "page": 1, "chunk_index": 0},
        )
        mock_vs.similarity_search_with_score.return_value = [(mock_doc, 0.5)]
        mock_get_vs.return_value = mock_vs

        results = query_documents("solar panels")
        assert len(results) == 1
        assert results[0]["text"] == "Solar panels are efficient."
        assert results[0]["metadata"]["source"] == "test.pdf"
        assert results[0]["score"] == 0.5

    @patch("src.vectorstore.get_vectorstore")
    def test_returns_empty_on_error(self, mock_get_vs):
        """Should return empty list on exception, not crash."""
        mock_get_vs.side_effect = Exception("Connection failed")
        results = query_documents("test question")
        assert results == []


class TestRagQuery:
    """Tests for the rag_query function."""

    @patch("src.vectorstore.query_documents")
    def test_no_results_returns_fallback_message(self, mock_query):
        """Should return a 'no info' message when no chunks are retrieved."""
        mock_query.return_value = []
        result = rag_query("What is solar energy?")
        assert "don't have enough information" in result["answer"]
        assert result["sources"] == []

    @patch("src.vectorstore.ChatGoogleGenerativeAI")
    @patch("src.vectorstore._get_api_key")
    @patch("src.vectorstore.query_documents")
    def test_returns_answer_with_sources(self, mock_query, mock_api_key, mock_llm_cls):
        """Should return an answer and source citations when chunks exist."""
        mock_query.return_value = [
            {
                "text": "Solar panels convert sunlight to electricity.",
                "metadata": {"source": "energy.pdf", "page": 2, "chunk_index": 0},
                "score": 0.3,
            }
        ]
        mock_api_key.return_value = "test-key"

        # Mock the LLM chain
        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Solar panels work by converting sunlight."
        mock_llm.__or__ = MagicMock(return_value=MagicMock(
            invoke=MagicMock(return_value=mock_response)
        ))
        mock_llm_cls.return_value = mock_llm

        result = rag_query("How do solar panels work?")
        assert "sources" in result
        assert len(result["sources"]) == 1
        assert result["sources"][0]["document"] == "energy.pdf"
        assert result["sources"][0]["page"] == 2

    @patch("src.vectorstore.ChatGoogleGenerativeAI")
    @patch("src.vectorstore._get_api_key")
    @patch("src.vectorstore.query_documents")
    def test_handles_llm_error_gracefully(self, mock_query, mock_api_key, mock_llm_cls):
        """Should return error info when LLM call fails."""
        mock_query.return_value = [
            {
                "text": "Some content",
                "metadata": {"source": "doc.pdf", "page": 1},
                "score": 0.4,
            }
        ]
        mock_api_key.return_value = "test-key"
        mock_llm_cls.side_effect = Exception("API quota exceeded")

        result = rag_query("test question")
        assert "error" in result or "error" in result.get("answer", "").lower()


class TestIngestAndStore:
    """Tests for the ingest_and_store convenience function."""

    @patch("src.vectorstore.add_documents")
    @patch("src.vectorstore.ingest_pdf")
    def test_successful_ingestion(self, mock_ingest, mock_add):
        """Should return chunk count on success."""
        mock_docs = [
            Document(page_content="chunk1", metadata={"page": 1}),
            Document(page_content="chunk2", metadata={"page": 1}),
        ]
        mock_ingest.return_value = mock_docs
        mock_add.return_value = MagicMock()

        result = ingest_and_store("data/sample.pdf")
        assert result["num_chunks"] == 2
        assert result["source"] == "sample.pdf"

    def test_file_not_found(self):
        """Should return error dict for missing file."""
        result = ingest_and_store("/nonexistent/path/file.pdf")
        assert "error" in result

    @patch("src.vectorstore.add_documents")
    @patch("src.vectorstore.ingest_pdf")
    def test_handles_embedding_error(self, mock_ingest, mock_add):
        """Should return error dict when embedding fails."""
        mock_ingest.return_value = [Document(page_content="test")]
        mock_add.side_effect = Exception("Embedding API error")

        result = ingest_and_store("data/sample.pdf")
        assert "error" in result
