# AI Document Assistant

A portfolio-quality, RAG-based document chat application powered by a LangGraph agentic router, equipped with safety layers and evaluation metrics.

## What It Does
The AI Document Assistant allows users to upload PDF documents and ask questions about them. The system uses an agentic router to intelligently classify user intents and route queries to the appropriate tools. It can:
- **Retrieve information** directly from the uploaded PDFs with precise page-level citations.
- **Search the web** using the Tavily API for current events not covered in the documents.
- **Perform calculations** safely via a built-in math tool using Python's AST parser.
- **Answer directly** for conversational queries.

It is wrapped in a strict safety layer that neutralizes prompt injection attempts and redacts Personally Identifiable Information (PII) from model outputs.

## Architecture

```mermaid
graph TD
    A[User Input] --> B[Input Sanitization]
    B -->|Cleaned Input| C{Intent Classifier}
    
    C -->|retrieve| D[retrieve_docs Tool]
    C -->|web_search| E[web_search Tool]
    C -->|calculate| F[calculator Tool]
    C -->|direct| G[Direct Answer]
    
    D --> H[ChromaDB Vector Store]
    E --> I[Tavily API]
    
    H --> J[Agent Graph]
    I --> J
    F --> J
    G --> J
    
    J --> K[Output PII Redaction]
    K --> L[Streamlit Frontend]
```

## Tech Stack
- **Language:** Python 3.11+
- **LLM & Embeddings:** Google Gemini (`gemini-2.5-flash`, `gemini-embedding-2`) via `langchain-google-genai`
- **Orchestration:** LangChain + LangGraph
- **Vector Store:** ChromaDB (Persisted)
- **Web Search:** Tavily API
- **Frontend:** Streamlit
- **Evaluation:** RAGAS (Manual fallback supported)
- **Testing & Linting:** pytest, ruff

## Setup & Run Instructions

1. **Clone the repository:**
   ```bash
   git clone <repo-url>
   cd ai-document-assistant
   ```

2. **Create a virtual environment and install dependencies:**
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

3. **Configure Environment Variables:**
   Copy the example file and add your API keys:
   ```bash
   cp .env.example .env
   ```
   *Required:* `GOOGLE_API_KEY` (Get one at [Google AI Studio](https://aistudio.google.com/))
   *Optional:* `TAVILY_API_KEY` (Required for web search functionality)

4. **Run the application:**
   ```bash
   streamlit run app.py
   ```

## Evaluation Results

Evaluation was performed using `src.eval` against a set of 10 sample Q&A pairs from the `eval/qa_pairs.json` file.
RAGAS requires OpenAI credentials for its LLM-as-a-judge scoring, so the evaluation uses a manual heuristic fallback scorer that measures keyword overlap for faithfulness, relevancy, and precision.

| Metric | Score | Description |
|--------|-------|-------------|
| **Faithfulness** | 0.9615 | Answer grounded in retrieved context |
| **Answer Relevancy** | 0.4661 | Answer addresses the question asked |
| **Context Precision** | 0.9178 | Most relevant chunks ranked highest |

9 out of 10 questions were answered correctly with source citations. The one question outside the document scope ("What is the capital of France?") was correctly refused with "I don't have enough information." The moderate answer relevancy score reflects the heuristic scorer's keyword-matching limitations rather than actual answer quality.

Full per-question results are in [`eval/eval_results.md`](eval/eval_results.md).

## Deployment

**Local deployment:**
```bash
streamlit run app.py
```
The app runs locally at `http://localhost:8501`.

**Streamlit Community Cloud deployment:**
1. Push this repository to GitHub.
2. Log in to [Streamlit Community Cloud](https://share.streamlit.io/).
3. Click "New app", select your repository, branch, and set the main file path to `app.py`.
4. In "Advanced settings", add your `.env` variables (`GOOGLE_API_KEY`, etc.).
5. Click "Deploy".

