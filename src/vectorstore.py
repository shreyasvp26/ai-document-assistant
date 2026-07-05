"""Vector store module — ChromaDB wrapper for document storage and retrieval.

This module provides a clean interface to ChromaDB for:
1. Storing document chunks with their embeddings
2. Querying for relevant chunks given a user question
3. Running the full RAG query pipeline (retrieve → prompt → generate → cite)

Design decisions:
- Uses GoogleGenerativeAIEmbeddings from langchain-google-genai
- Model names read from env vars (GEMINI_MODEL, GEMINI_EMBEDDING_MODEL)
- Chroma persisted to ./chroma_db for local development
- RAG prompt explicitly instructs model to cite sources and admit uncertainty
"""

import os
import logging
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from src.ingestion import ingest_pdf

load_dotenv()
logger = logging.getLogger(__name__)

# --- Configuration ---
CHROMA_PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = "documents"
DEFAULT_TOP_K = 5


def _get_api_key() -> str:
    """Validate and return the Google API key from environment.

    Returns:
        The GOOGLE_API_KEY value.

    Raises:
        EnvironmentError: If GOOGLE_API_KEY is not set.
    """
    key = os.getenv("GOOGLE_API_KEY")
    if not key:
        raise EnvironmentError(
            "GOOGLE_API_KEY not set. "
            "Copy .env.example to .env and add your key. "
            "Get one at https://aistudio.google.com/app/apikey"
        )
    return key


def get_embedding_function() -> GoogleGenerativeAIEmbeddings:
    """Create and return the embedding function using the configured Gemini model.

    Reads the embedding model name from GEMINI_EMBEDDING_MODEL env var,
    defaulting to 'gemini-embedding-2' if not set.

    Returns:
        GoogleGenerativeAIEmbeddings instance ready for use.
    """
    _get_api_key()  # Validate key exists before creating embeddings
    model_name = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")
    logger.info(f"Using embedding model: {model_name}")

    return GoogleGenerativeAIEmbeddings(
        model=f"models/{model_name}",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
    )


def get_vectorstore(
    persist_directory: str = CHROMA_PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
) -> Chroma:
    """Get or create a Chroma vector store instance.

    If a persisted store exists at the given directory, it will be loaded.
    Otherwise, a new empty store is created.

    Args:
        persist_directory: Path to the Chroma persistence directory.
        collection_name: Name of the Chroma collection.

    Returns:
        Chroma vector store instance.
    """
    embedding_fn = get_embedding_function()

    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding_fn,
        persist_directory=persist_directory,
    )


def add_documents(
    documents: list[Document],
    persist_directory: str = CHROMA_PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
) -> Chroma:
    """Add document chunks to the vector store, embedding them automatically.

    Args:
        documents: List of Document objects to store (typically from ingest_pdf).
        persist_directory: Path to persist the Chroma database.
        collection_name: Name of the Chroma collection.

    Returns:
        The Chroma vector store instance with documents added.

    Raises:
        ValueError: If no documents are provided.
        Exception: If embedding or storage fails (logged and re-raised).
    """
    if not documents:
        raise ValueError("No documents provided to add to the vector store.")

    try:
        embedding_fn = get_embedding_function()

        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embedding_fn,
            persist_directory=persist_directory,
            collection_name=collection_name,
        )

        logger.info(
            f"Added {len(documents)} documents to Chroma "
            f"(collection='{collection_name}', dir='{persist_directory}')"
        )
        return vectorstore

    except Exception as e:
        logger.error(f"Failed to add documents to vector store: {e}")
        raise


def query_documents(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    persist_directory: str = CHROMA_PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
) -> list[dict]:
    """Query the vector store for chunks relevant to the given question.

    Args:
        question: The user's question to search for.
        top_k: Number of most relevant chunks to return.
        persist_directory: Path to the Chroma persistence directory.
        collection_name: Name of the Chroma collection.

    Returns:
        List of dicts, each containing:
        - 'text': The chunk's text content
        - 'metadata': Dict with 'source', 'page', 'chunk_index'
        - 'score': Similarity score (lower = more similar for Chroma's L2 distance)
    """
    try:
        vectorstore = get_vectorstore(persist_directory, collection_name)
        results = vectorstore.similarity_search_with_score(question, k=top_k)

        formatted = []
        for doc, score in results:
            formatted.append({
                "text": doc.page_content,
                "metadata": doc.metadata,
                "score": float(score),
            })

        logger.info(
            f"Query '{question[:50]}...' returned {len(formatted)} results"
        )
        return formatted

    except Exception as e:
        logger.error(f"Vector store query failed: {e}")
        return []


# --- RAG Query Pipeline (Phase 2) ---

# System prompt that enforces grounded answers with citations
RAG_SYSTEM_PROMPT = """You are a helpful document assistant. Answer the user's question 
based ONLY on the provided context from uploaded documents.

RULES:
1. Only use information from the provided context to answer.
2. If the context does not contain enough information to answer the question, 
   say: "I don't have enough information in the uploaded documents to answer this question."
3. Do NOT make up information or use your general knowledge.
4. Always cite which document and page number(s) your answer comes from.
5. Be concise but thorough.

CONTEXT FROM DOCUMENTS:
{context}
"""

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", RAG_SYSTEM_PROMPT),
    ("human", "{question}"),
])


def rag_query(
    question: str,
    top_k: int = DEFAULT_TOP_K,
    persist_directory: str = CHROMA_PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
) -> dict:
    """Run the full RAG pipeline: retrieve relevant chunks → generate grounded answer.

    This is the main entry point for answering questions about uploaded documents.
    The pipeline:
    1. Queries the vector store for relevant chunks
    2. Builds a grounded prompt with the retrieved context
    3. Calls the Gemini model to generate an answer
    4. Returns the answer along with source citations

    Args:
        question: The user's natural language question.
        top_k: Number of chunks to retrieve for context.
        persist_directory: Path to the Chroma persistence directory.
        collection_name: Name of the Chroma collection.

    Returns:
        Dict with keys:
        - 'answer': The generated answer text
        - 'sources': List of source dicts with 'document', 'page', 'chunk' keys
        - 'error': Error message string (only present if something failed)
    """
    # Step 1: Retrieve relevant chunks
    retrieved = query_documents(question, top_k, persist_directory, collection_name)

    if not retrieved:
        return {
            "answer": (
                "I don't have enough information in the uploaded documents "
                "to answer this question. Please upload a relevant document first."
            ),
            "sources": [],
        }

    # Step 2: Build context string from retrieved chunks
    context_parts = []
    sources = []
    for i, chunk in enumerate(retrieved):
        source_info = {
            "document": chunk["metadata"].get("source", "Unknown"),
            "page": chunk["metadata"].get("page", "?"),
            "chunk": chunk["text"][:300],  # Truncate for display
        }
        sources.append(source_info)

        # Format context with source attribution for the prompt
        context_parts.append(
            f"[Source: {source_info['document']}, Page {source_info['page']}]\n"
            f"{chunk['text']}"
        )

    context = "\n\n---\n\n".join(context_parts)

    # Step 3: Generate answer with Gemini
    try:
        _get_api_key()
        model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.1,  # Low temperature for factual, grounded answers
        )

        chain = RAG_PROMPT | llm
        response = chain.invoke({"context": context, "question": question})

        return {
            "answer": response.content,
            "sources": sources,
        }

    except Exception as e:
        logger.error(f"RAG query failed during generation: {e}")
        return {
            "answer": f"Sorry, I encountered an error generating a response: {str(e)}",
            "sources": sources,
            "error": str(e),
        }


def ingest_and_store(
    file_path: str,
    persist_directory: str = CHROMA_PERSIST_DIR,
    collection_name: str = COLLECTION_NAME,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> dict:
    """Full pipeline: load PDF → chunk → embed → store in Chroma.

    Convenience function that combines ingestion and storage into one call.

    Args:
        file_path: Path to the PDF file.
        persist_directory: Where to persist the Chroma database.
        collection_name: Chroma collection name.
        chunk_size: Characters per chunk.
        chunk_overlap: Overlap between chunks.

    Returns:
        Dict with 'num_chunks' on success, or 'error' on failure.
    """
    try:
        chunks = ingest_pdf(file_path, chunk_size, chunk_overlap)
        add_documents(chunks, persist_directory, collection_name)
        return {
            "num_chunks": len(chunks),
            "source": Path(file_path).name,
        }
    except (FileNotFoundError, ValueError) as e:
        return {"error": str(e)}
    except Exception as e:
        logger.error(f"Ingestion pipeline failed: {e}")
        return {"error": f"Failed to process document: {str(e)}"}


# --- CLI entry point for standalone testing ---
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python -m src.vectorstore <command> [args]")
        print("Commands:")
        print("  ingest <pdf_path>    - Ingest a PDF into the vector store")
        print("  query <question>     - Query the vector store")
        print("  rag <question>       - Full RAG query with answer generation")
        sys.exit(1)

    command = sys.argv[1]

    if command == "ingest":
        pdf_path = sys.argv[2] if len(sys.argv) > 2 else "data/sample.pdf"
        result = ingest_and_store(pdf_path)
        if "error" in result:
            print(f"❌ {result['error']}")
        else:
            print(f"✅ Ingested {result['num_chunks']} chunks from '{result['source']}'")

    elif command == "query":
        question = " ".join(sys.argv[2:])
        results = query_documents(question)
        print(f"\n🔍 Query: '{question}'")
        print(f"   Found {len(results)} results:\n")
        for i, r in enumerate(results):
            print(f"   [{i+1}] Page {r['metadata'].get('page', '?')} "
                  f"(score: {r['score']:.4f})")
            print(f"       {r['text'][:150]}...\n")

    elif command == "rag":
        question = " ".join(sys.argv[2:])
        result = rag_query(question)
        print(f"\n🤖 Question: '{question}'")
        print(f"\n📝 Answer: {result['answer']}")
        if result["sources"]:
            print("\n📄 Sources:")
            for s in result["sources"]:
                print(f"   - {s['document']}, Page {s['page']}")
    else:
        print(f"Unknown command: {command}")
