# AI Document Assistant — Project Guide

## Tech Stack

| Layer | Technology | Version |
|-------|-----------|---------|
| Language | Python | 3.11+ |
| LLM | Google Gemini (`gemini-2.5-flash`) | via `langchain-google-genai` |
| Embeddings | Gemini Embedding 2 (`gemini-embedding-2`) | via `langchain-google-genai` |
| Orchestration | LangChain + LangGraph | 0.3.x / 0.2.x |
| Vector Store | ChromaDB (persisted to `./chroma_db`) | 1.5.x |
| Web Search | Tavily API (free tier) | `tavily-python` |
| Frontend | Streamlit | 1.58+ |
| Evaluation | RAGAS | 0.4.x |
| Testing | pytest (mock all API calls) | 9.x |
| Linting | ruff | 0.15.x |

## Module Ownership

```
src/
├── ingestion.py      # PDF loading + chunking (PyPDF → RecursiveCharacterTextSplitter)
├── vectorstore.py    # Chroma wrapper: add_documents, query_documents, rag_query
├── tools.py          # Three LangChain @tool defs: retrieve_docs, web_search, calculator
├── agent.py          # LangGraph StateGraph: classify_intent → route → handle
├── safety.py         # Input sanitization (prompt injection) + output PII redaction
└── eval.py           # RAGAS evaluation: loads qa_pairs.json → scores → eval_results.md
```

### Dependency Graph

```
ingestion.py       → (standalone)
vectorstore.py     → ingestion.py
tools.py           → vectorstore.py
safety.py          → (standalone)
agent.py           → tools.py, safety.py, vectorstore.py
eval.py            → vectorstore.py
app.py             → agent.py, vectorstore.py, safety.py
```

## Environment Variables

| Variable | Required | Default | Where Read |
|----------|----------|---------|------------|
| `GOOGLE_API_KEY` | ✅ | — | vectorstore.py, agent.py |
| `GEMINI_MODEL` | ❌ | `gemini-2.5-flash` | vectorstore.py, agent.py |
| `GEMINI_EMBEDDING_MODEL` | ❌ | `gemini-embedding-2` | vectorstore.py |
| `TAVILY_API_KEY` | ❌ | — | tools.py (web search disabled if missing) |
| `CHROMA_PERSIST_DIR` | ❌ | `./chroma_db` | vectorstore.py |

If a required var is missing → fail on startup with a message naming the exact key to set.

## Error-Handling Conventions

**Every external call** (Gemini, Tavily, file I/O, ChromaDB) is wrapped in try/except:

```python
try:
    result = external_api.call(...)
except SpecificError as e:
    logger.error(f"Context: {e}")
    return {"error": "User-facing message explaining what went wrong."}
except Exception as e:
    logger.error(f"Unexpected: {e}")
    return {"error": f"An unexpected error occurred: {str(e)}"}
```

Rules:
1. Never crash the Streamlit app — always return an error dict or display `st.error()`.
2. Log the technical error with `logger.error()` for debugging.
3. Return a **user-facing** message (no raw stack traces).
4. Functions that can fail return `{"result": ..., "sources": [...]}` on success, `{"error": "..."}` on failure.

## RAG Answer Rules

1. Every answer **must cite** which document and page the information came from.
2. If retrieved context doesn't support an answer → say "I don't have enough information" — never guess.
3. Safety pipeline applied at both ends: `sanitize_input()` before the model, `redact_pii()` after.

## Testing Conventions

**Mock strategy:**
- ✅ Always mock: Gemini API calls, Tavily API calls (no network, no quota burn)
- ❌ Never mock: calculator logic, safety regex, ingestion chunking (pure logic)

**Test structure:**
```python
class TestFunctionName:
    """Tests for function_name in src/module.py."""
    def test_happy_path(self): ...
    def test_edge_case(self): ...
    def test_error_handling(self): ...
```

**Shared fixtures** in `tests/conftest.py`:
- `sample_pdf_path` — path to `data/sample.pdf`
- `sample_text_chunks` — pre-built chunk dicts
- `mock_llm_response` — mocked Gemini response
- `mock_embedding` — 768-dim vector

**Run:** `pytest tests/ -v --tb=short`

## Docstring Rules

Every function gets a docstring. Non-obvious logic gets inline comments.
An interviewer will read this repo — write for readability.

```python
def my_function(input: str) -> dict:
    """One-line summary.

    Detailed explanation if non-obvious.

    Args:
        input: What this parameter is.

    Returns:
        Dict with 'result' or 'error' key.
    """
```

## Git Conventions

One commit per completed phase:
```
phase(0): scaffolding and infrastructure
phase(1): ingestion and vector store
phase(2): rag query pipeline
phase(3): tools
phase(4): langgraph agent
phase(5): safety and evaluation
phase(6): streamlit frontend
phase(7): tests
phase(8): readme and deployment
```

## Security Summary

See `SECURITY.md` for full details. Key points:
- No hardcoded API keys anywhere — `.env` is gitignored
- Prompt injection defense: 9 regex patterns in `safety.py`
- PII redaction: emails + phone numbers masked in model output
- Calculator: AST-based parsing, no `eval()` on raw strings
- Pre-commit hooks scan for secrets before they reach git
