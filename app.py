"""AI Document Assistant — Streamlit Frontend.

A RAG-based document chat application with:
- PDF upload and ingestion into ChromaDB
- Chat interface with source citations
- Document summarization
- Conversation export to markdown
- Agentic routing (doc search, web search, calculator, general chat)

All errors are caught and displayed gracefully — nothing crashes the app.
"""

import os
import sys
import tempfile
import logging
from datetime import datetime
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

# Ensure the project root is in the Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Page Configuration ---
st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Custom CSS for Premium Look ---
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main header styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
    }

    .main-header h1 {
        margin: 0;
        font-size: 1.8rem;
        font-weight: 700;
        color: white;
    }

    .main-header p {
        margin: 0.3rem 0 0 0;
        font-size: 0.95rem;
        opacity: 0.9;
        color: rgba(255, 255, 255, 0.9);
    }

    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 500;
        margin-right: 0.5rem;
    }

    .badge-success {
        background: rgba(34, 197, 94, 0.15);
        color: #16a34a;
        border: 1px solid rgba(34, 197, 94, 0.3);
    }

    .badge-warning {
        background: rgba(234, 179, 8, 0.15);
        color: #ca8a04;
        border: 1px solid rgba(234, 179, 8, 0.3);
    }

    .badge-info {
        background: rgba(59, 130, 246, 0.15);
        color: #2563eb;
        border: 1px solid rgba(59, 130, 246, 0.3);
    }

    /* Source citation cards */
    .source-card {
        background: #f8f9fc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.75rem 1rem;
        margin: 0.5rem 0;
        font-size: 0.85rem;
    }

    .source-card .source-header {
        font-weight: 600;
        color: #4a5568;
        margin-bottom: 0.25rem;
    }

    .source-card .source-text {
        color: #718096;
        font-size: 0.8rem;
        line-height: 1.4;
    }

    /* Sidebar styling */
    .sidebar-section {
        background: #f8f9fc;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid #e2e8f0;
    }

    /* Tool badge */
    .tool-badge {
        display: inline-block;
        padding: 0.15rem 0.5rem;
        border-radius: 12px;
        font-size: 0.7rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .tool-retrieve { background: #dbeafe; color: #1d4ed8; }
    .tool-web_search { background: #dcfce7; color: #15803d; }
    .tool-calculator { background: #fef3c7; color: #92400e; }
    .tool-direct { background: #f3e8ff; color: #7c3aed; }
</style>
""", unsafe_allow_html=True)


# --- Startup Validation ---
def validate_environment() -> list[str]:
    """Check that required environment variables are set.

    Returns:
        List of warning messages for missing optional vars.
        Raises an error in the UI for missing required vars.
    """
    warnings = []

    if not os.getenv("GOOGLE_API_KEY"):
        st.error(
            "🔑 **GOOGLE_API_KEY is not set!**\n\n"
            "This is required to use the AI Document Assistant.\n\n"
            "1. Get a free API key at [Google AI Studio](https://aistudio.google.com/app/apikey)\n"
            "2. Add it to your `.env` file: `GOOGLE_API_KEY=your-key-here`\n"
            "3. Restart the app"
        )
        st.stop()

    if not os.getenv("TAVILY_API_KEY"):
        warnings.append(
            "⚠️ TAVILY_API_KEY not set — web search is disabled. "
            "Get a free key at [tavily.com](https://tavily.com)"
        )

    return warnings


# --- Session State Initialization ---
def init_session_state():
    """Initialize Streamlit session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "documents_ingested" not in st.session_state:
        st.session_state.documents_ingested = False
    if "document_name" not in st.session_state:
        st.session_state.document_name = None
    if "num_chunks" not in st.session_state:
        st.session_state.num_chunks = 0
    if "chroma_dir" not in st.session_state:
        # Use a session-specific temp directory for Chroma
        st.session_state.chroma_dir = tempfile.mkdtemp(prefix="chroma_")


# --- Sidebar ---
def render_sidebar():
    """Render the sidebar with upload, status, and actions."""
    with st.sidebar:
        st.markdown("### 📄 Document Upload")

        uploaded_file = st.file_uploader(
            "Upload a PDF document",
            type=["pdf"],
            help="Upload a PDF to ask questions about its content.",
        )

        if uploaded_file is not None:
            if (
                not st.session_state.documents_ingested
                or st.session_state.document_name != uploaded_file.name
            ):
                with st.spinner("📥 Processing document..."):
                    try:
                        ingest_uploaded_file(uploaded_file)
                    except Exception as e:
                        st.error(f"❌ Failed to process document: {str(e)}")

        # Status section
        st.markdown("---")
        st.markdown("### 📊 Status")

        if st.session_state.documents_ingested:
            st.markdown(
                f'<span class="status-badge badge-success">✅ Document Loaded</span>',
                unsafe_allow_html=True,
            )
            st.markdown(f"**Document:** {st.session_state.document_name}")
            st.markdown(f"**Chunks:** {st.session_state.num_chunks}")
        else:
            st.markdown(
                '<span class="status-badge badge-warning">⏳ No document uploaded</span>',
                unsafe_allow_html=True,
            )

        # Actions section
        st.markdown("---")
        st.markdown("### ⚡ Quick Actions")

        col1, col2 = st.columns(2)
        with col1:
            if st.button("📝 Summarize", use_container_width=True):
                if st.session_state.documents_ingested:
                    summarize_document()
                else:
                    st.warning("Please upload a document first.")

        with col2:
            if st.button("💾 Save Notes", use_container_width=True):
                export_conversation()

        # Info section
        st.markdown("---")
        st.markdown("### ℹ️ Capabilities")
        st.markdown("""
        - 📄 **Ask about documents** — upload a PDF and chat
        - 🌐 **Web search** — ask about current events
        - 🔢 **Calculator** — math expressions
        - 💬 **General chat** — greetings and help
        """)


def ingest_uploaded_file(uploaded_file) -> None:
    """Process an uploaded PDF file and store it in the vector store.

    Args:
        uploaded_file: Streamlit UploadedFile object.
    """
    from src.vectorstore import ingest_and_store

    # Save uploaded file to a temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        result = ingest_and_store(
            tmp_path,
            persist_directory=st.session_state.chroma_dir,
        )

        if "error" in result:
            st.error(f"❌ {result['error']}")
            return

        st.session_state.documents_ingested = True
        st.session_state.document_name = uploaded_file.name
        st.session_state.num_chunks = result["num_chunks"]

        st.success(
            f"✅ Processed **{uploaded_file.name}** — "
            f"{result['num_chunks']} chunks indexed"
        )

    except Exception as e:
        st.error(f"❌ Failed to process document: {str(e)}")
    finally:
        # Clean up temp file
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def summarize_document():
    """Generate a summary of the uploaded document."""
    if not st.session_state.documents_ingested:
        return

    summary_prompt = (
        "Please provide a comprehensive summary of the uploaded document. "
        "Cover the main topics, key findings, and important details."
    )

    # Add user message
    st.session_state.messages.append({
        "role": "user",
        "content": "📝 Summarize this document",
        "tool_used": None,
        "sources": None,
    })

    # Generate summary
    try:
        from src.vectorstore import rag_query

        result = rag_query(
            summary_prompt,
            top_k=10,  # Get more chunks for a comprehensive summary
            persist_directory=st.session_state.chroma_dir,
        )

        from src.safety import apply_output_safety
        answer = apply_output_safety(result["answer"])

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "tool_used": "retrieve_docs",
            "sources": result.get("sources", []),
        })

    except Exception as e:
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"Sorry, I couldn't generate a summary: {str(e)}",
            "tool_used": "error",
            "sources": None,
        })

    st.rerun()


def export_conversation():
    """Export the conversation history as a downloadable markdown file."""
    if not st.session_state.messages:
        st.sidebar.warning("No conversation to export yet.")
        return

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    doc_name = st.session_state.document_name or "no-document"

    lines = [
        f"# AI Document Assistant — Conversation Notes",
        f"",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"**Document:** {doc_name}",
        f"",
        f"---",
        f"",
    ]

    for msg in st.session_state.messages:
        role = "🧑 User" if msg["role"] == "user" else "🤖 Assistant"
        lines.append(f"### {role}")
        lines.append(f"")
        lines.append(msg["content"])
        lines.append(f"")

        if msg.get("sources"):
            lines.append("**Sources:**")
            seen = set()
            for s in msg["sources"]:
                key = f"{s.get('document', '?')}-p{s.get('page', '?')}"
                if key not in seen:
                    lines.append(f"- {s.get('document', '?')}, Page {s.get('page', '?')}")
                    seen.add(key)
            lines.append("")

        lines.append("---")
        lines.append("")

    content = "\n".join(lines)

    st.sidebar.download_button(
        label="📥 Download Notes",
        data=content,
        file_name=f"notes_{timestamp}.md",
        mime="text/markdown",
    )


def render_tool_badge(tool_used: str) -> str:
    """Return HTML for a tool badge based on the tool name."""
    tool_labels = {
        "retrieve_docs": ("📄 Document Search", "tool-retrieve"),
        "web_search": ("🌐 Web Search", "tool-web_search"),
        "calculator": ("🔢 Calculator", "tool-calculator"),
        "direct_answer": ("💬 Direct Answer", "tool-direct"),
    }
    label, css_class = tool_labels.get(tool_used, ("🔧 Tool", "tool-direct"))
    return f'<span class="tool-badge {css_class}">{label}</span>'


def render_sources(sources: list):
    """Render source citations in an expandable section."""
    if not sources:
        return

    with st.expander("📄 Sources & Citations", expanded=False):
        seen = set()
        for source in sources:
            doc = source.get("document", "Unknown")
            page = source.get("page", "?")
            chunk = source.get("chunk", "")
            key = f"{doc}-p{page}"

            if key not in seen:
                seen.add(key)
                st.markdown(f"""
                <div class="source-card">
                    <div class="source-header">📄 {doc} — Page {page}</div>
                    <div class="source-text">{chunk[:250]}...</div>
                </div>
                """, unsafe_allow_html=True)


# --- Main Chat Interface ---
def main():
    """Main application entry point."""
    init_session_state()

    # Validate environment
    env_warnings = validate_environment()

    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📄 AI Document Assistant</h1>
        <p>Upload PDFs, ask questions, and get AI-powered answers with source citations</p>
    </div>
    """, unsafe_allow_html=True)

    # Show environment warnings
    for warning in env_warnings:
        st.warning(warning)

    # Render sidebar
    render_sidebar()

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            # Show tool badge for assistant messages
            if message["role"] == "assistant" and message.get("tool_used"):
                st.markdown(
                    render_tool_badge(message["tool_used"]),
                    unsafe_allow_html=True,
                )

            st.markdown(message["content"])

            # Show sources for assistant messages
            if message["role"] == "assistant" and message.get("sources"):
                render_sources(message["sources"])

    # Chat input
    if user_input := st.chat_input("Ask a question about your documents, the web, or math..."):
        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)

        st.session_state.messages.append({
            "role": "user",
            "content": user_input,
            "tool_used": None,
            "sources": None,
        })

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    from src.agent import run_agent

                    # Override the chroma dir for the agent
                    os.environ["CHROMA_PERSIST_DIR"] = st.session_state.chroma_dir

                    result = run_agent(user_input)

                    # Show tool badge
                    tool_used = result.get("tool_used", "")
                    if tool_used:
                        st.markdown(
                            render_tool_badge(tool_used),
                            unsafe_allow_html=True,
                        )

                    # Show response
                    st.markdown(result["response"])

                    # Show sources
                    sources = result.get("sources", [])
                    if sources:
                        render_sources(sources)

                    # Store in history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": result["response"],
                        "tool_used": tool_used,
                        "sources": sources,
                    })

                except Exception as e:
                    error_msg = f"Sorry, an error occurred: {str(e)}"
                    st.error(error_msg)
                    logger.error(f"Chat error: {e}", exc_info=True)

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg,
                        "tool_used": "error",
                        "sources": None,
                    })


if __name__ == "__main__":
    main()
