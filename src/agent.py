"""LangGraph agent — routes user queries to the appropriate tool or direct answer.

This module implements a stateful agent using LangGraph that:
1. Classifies user intent (document question, web search, math, general chat)
2. Routes to the appropriate tool (retrieve_docs, web_search, calculator)
3. Falls back to a direct answer for general conversation
4. Logs all routing decisions for transparency

Architecture:
    User Input → classify_intent → route → [retrieve | search | calculate | direct] → response

Design decisions:
- Uses Gemini for intent classification (lightweight, single-call classification)
- LangGraph StateGraph for explicit, debuggable control flow
- Safety layer applied at input and output boundaries
- All routing decisions logged to stdout for interview demo visibility
"""

import os
import logging
from typing import Literal, TypedDict, Annotated

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END

from src.tools import retrieve_docs, web_search, calculator, safe_calculate
from src.vectorstore import rag_query
from src.safety import apply_safety, apply_output_safety

load_dotenv()
logger = logging.getLogger(__name__)

# --- Intent Classification ---

INTENT_CLASSIFICATION_PROMPT = """You are an intent classifier for a document assistant chatbot.
Classify the user's message into exactly ONE of these categories:

1. "retrieve" — The user is asking a question about uploaded documents or PDFs.
   Examples: "What does the document say about...", "According to the paper...",
   "Summarize the document", "What is on page 3?"

2. "web_search" — The user needs current/real-time information not found in documents.
   Examples: "What's the current price of...", "Latest news about...",
   "Who won the election?", "What's the weather?"

3. "calculate" — The user wants to perform a mathematical calculation.
   Examples: "What is 15% of 200?", "Calculate 2^10", "Convert 100°F to Celsius"

4. "direct" — General conversation, greetings, or questions about your capabilities.
   Examples: "Hello", "What can you do?", "Thank you", "Help me understand..."

Respond with ONLY the category name (retrieve, web_search, calculate, or direct).
Do NOT include any explanation or additional text."""


class AgentState(TypedDict):
    """State schema for the LangGraph agent.

    Attributes:
        user_input: The original user message (after safety sanitization).
        intent: Classified intent (retrieve, web_search, calculate, direct).
        response: The generated response text.
        sources: List of source citation dicts (for RAG responses).
        tool_used: Name of the tool that was invoked.
    """
    user_input: str
    intent: str
    response: str
    sources: list
    tool_used: str


def _get_llm() -> ChatGoogleGenerativeAI:
    """Create a ChatGoogleGenerativeAI instance with configured model.

    Returns:
        Configured LLM instance.

    Raises:
        EnvironmentError: If GOOGLE_API_KEY is not set.
    """
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "GOOGLE_API_KEY not set. "
            "Copy .env.example to .env and add your key."
        )

    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    return ChatGoogleGenerativeAI(
        model=model_name,
        google_api_key=api_key,
        temperature=0.1,
    )


# --- Graph Nodes ---

def classify_intent(state: AgentState) -> AgentState:
    """Classify the user's intent to determine which tool to use.

    Uses Gemini to classify the input into one of four categories:
    retrieve, web_search, calculate, or direct.

    Args:
        state: Current agent state with user_input populated.

    Returns:
        Updated state with intent field set.
    """
    user_input = state["user_input"]
    logger.info(f"[ROUTER] Classifying intent for: '{user_input[:80]}...'")

    try:
        llm = _get_llm()
        messages = [
            SystemMessage(content=INTENT_CLASSIFICATION_PROMPT),
            HumanMessage(content=user_input),
        ]
        response = llm.invoke(messages)
        intent = response.content.strip().lower()

        # Validate the intent is one of the expected values
        valid_intents = {"retrieve", "web_search", "calculate", "direct"}
        if intent not in valid_intents:
            logger.warning(
                f"[ROUTER] Unexpected intent '{intent}', defaulting to 'direct'"
            )
            intent = "direct"

        logger.info(f"[ROUTER] ✅ Intent classified as: '{intent}'")
        print(f"\n🔀 ROUTING DECISION: '{user_input[:60]}...' → {intent.upper()}")

    except Exception as e:
        logger.error(f"[ROUTER] Intent classification failed: {e}")
        intent = "direct"  # Safe fallback
        print(f"\n⚠️ ROUTING FALLBACK: Classification failed, using DIRECT")

    return {**state, "intent": intent}


def handle_retrieve(state: AgentState) -> AgentState:
    """Handle document retrieval queries using the RAG pipeline.

    Args:
        state: Agent state with user_input and intent='retrieve'.

    Returns:
        Updated state with response and sources.
    """
    user_input = state["user_input"]
    logger.info(f"[RETRIEVE] Searching documents for: '{user_input[:80]}...'")
    print(f"📄 TOOL: retrieve_docs — searching uploaded documents...")

    try:
        result = rag_query(user_input)
        response = apply_output_safety(result["answer"])
        sources = result.get("sources", [])

        return {
            **state,
            "response": response,
            "sources": sources,
            "tool_used": "retrieve_docs",
        }

    except Exception as e:
        logger.error(f"[RETRIEVE] Failed: {e}")
        return {
            **state,
            "response": f"Sorry, I encountered an error searching the documents: {str(e)}",
            "sources": [],
            "tool_used": "retrieve_docs",
        }


def handle_web_search(state: AgentState) -> AgentState:
    """Handle web search queries using Tavily.

    Args:
        state: Agent state with user_input and intent='web_search'.

    Returns:
        Updated state with web search results.
    """
    user_input = state["user_input"]
    logger.info(f"[WEB_SEARCH] Searching web for: '{user_input[:80]}...'")
    print(f"🌐 TOOL: web_search — searching the internet...")

    try:
        result = web_search.invoke(user_input)
        response = apply_output_safety(result)

        return {
            **state,
            "response": response,
            "sources": [],
            "tool_used": "web_search",
        }

    except Exception as e:
        logger.error(f"[WEB_SEARCH] Failed: {e}")
        return {
            **state,
            "response": f"Sorry, web search failed: {str(e)}",
            "sources": [],
            "tool_used": "web_search",
        }


def handle_calculate(state: AgentState) -> AgentState:
    """Handle mathematical calculation requests.

    Extracts the mathematical expression from the user's input and
    evaluates it using the safe calculator.

    Args:
        state: Agent state with user_input and intent='calculate'.

    Returns:
        Updated state with calculation result.
    """
    user_input = state["user_input"]
    logger.info(f"[CALCULATE] Processing: '{user_input[:80]}...'")
    print(f"🔢 TOOL: calculator — computing expression...")

    try:
        # Try to extract a mathematical expression from the input
        # First, try the raw input as an expression
        llm = _get_llm()
        extract_prompt = (
            "Extract ONLY the mathematical expression from this text. "
            "Return ONLY the expression with numbers and operators, nothing else. "
            "If the text describes a calculation in words, convert it to a math expression. "
            f"Examples: '15% of 200' → '200 * 15 / 100', 'square root of 144' → '144 ** 0.5'\n\n"
            f"Text: {user_input}"
        )
        expression_response = llm.invoke([HumanMessage(content=extract_prompt)])
        expression = expression_response.content.strip()

        result = safe_calculate(expression)

        if result.startswith("Error:"):
            response = f"I couldn't calculate that. {result}"
        else:
            response = f"**Calculation:** `{expression}` = **{result}**"

        return {
            **state,
            "response": apply_output_safety(response),
            "sources": [],
            "tool_used": "calculator",
        }

    except Exception as e:
        logger.error(f"[CALCULATE] Failed: {e}")
        return {
            **state,
            "response": f"Sorry, I couldn't perform that calculation: {str(e)}",
            "sources": [],
            "tool_used": "calculator",
        }


def handle_direct(state: AgentState) -> AgentState:
    """Handle general conversation without any tool use.

    Args:
        state: Agent state with user_input and intent='direct'.

    Returns:
        Updated state with direct LLM response.
    """
    user_input = state["user_input"]
    logger.info(f"[DIRECT] Generating direct response for: '{user_input[:80]}...'")
    print(f"💬 TOOL: direct_answer — responding directly...")

    try:
        llm = _get_llm()
        system_msg = (
            "You are a helpful AI document assistant. You can help users with:\n"
            "1. Answering questions about uploaded PDF documents\n"
            "2. Searching the web for current information\n"
            "3. Performing mathematical calculations\n"
            "4. General conversation\n\n"
            "Be concise, friendly, and helpful."
        )
        messages = [
            SystemMessage(content=system_msg),
            HumanMessage(content=user_input),
        ]
        response = llm.invoke(messages)

        return {
            **state,
            "response": apply_output_safety(response.content),
            "sources": [],
            "tool_used": "direct_answer",
        }

    except Exception as e:
        logger.error(f"[DIRECT] Failed: {e}")
        return {
            **state,
            "response": f"Sorry, I encountered an error: {str(e)}",
            "sources": [],
            "tool_used": "direct_answer",
        }


# --- Router Function ---

def route_by_intent(state: AgentState) -> str:
    """Route to the appropriate handler based on classified intent.

    This is the conditional edge function for the LangGraph graph.

    Args:
        state: Current agent state with intent field set.

    Returns:
        Name of the next node to execute.
    """
    intent = state.get("intent", "direct")
    route_map = {
        "retrieve": "handle_retrieve",
        "web_search": "handle_web_search",
        "calculate": "handle_calculate",
        "direct": "handle_direct",
    }
    return route_map.get(intent, "handle_direct")


# --- Graph Construction ---

def build_agent_graph() -> StateGraph:
    """Build and compile the LangGraph agent graph.

    Graph structure:
        classify_intent → (conditional) → [handle_retrieve | handle_web_search |
                                            handle_calculate | handle_direct] → END

    Returns:
        Compiled LangGraph graph ready for invocation.
    """
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("classify_intent", classify_intent)
    graph.add_node("handle_retrieve", handle_retrieve)
    graph.add_node("handle_web_search", handle_web_search)
    graph.add_node("handle_calculate", handle_calculate)
    graph.add_node("handle_direct", handle_direct)

    # Set entry point
    graph.set_entry_point("classify_intent")

    # Add conditional routing from classifier to handlers
    graph.add_conditional_edges(
        "classify_intent",
        route_by_intent,
        {
            "handle_retrieve": "handle_retrieve",
            "handle_web_search": "handle_web_search",
            "handle_calculate": "handle_calculate",
            "handle_direct": "handle_direct",
        },
    )

    # All handlers go to END
    graph.add_edge("handle_retrieve", END)
    graph.add_edge("handle_web_search", END)
    graph.add_edge("handle_calculate", END)
    graph.add_edge("handle_direct", END)

    return graph.compile()


# --- Main Entry Point ---

# Compile the graph once at module level for reuse
_agent_graph = None


def get_agent():
    """Get or create the compiled agent graph (lazy singleton).

    Returns:
        Compiled LangGraph agent.
    """
    global _agent_graph
    if _agent_graph is None:
        _agent_graph = build_agent_graph()
    return _agent_graph


def run_agent(user_input: str) -> dict:
    """Run the agent on a user input, returning the response and metadata.

    This is the main entry point for the agent. It:
    1. Applies input safety sanitization
    2. Classifies intent
    3. Routes to the appropriate tool
    4. Applies output safety (PII redaction)
    5. Returns the response with metadata

    Args:
        user_input: The user's raw message text.

    Returns:
        Dict with keys:
        - 'response': The generated response text (PII-redacted)
        - 'sources': List of source citation dicts (for RAG responses)
        - 'tool_used': Name of the tool that was invoked
        - 'intent': The classified intent
    """
    if not user_input or not user_input.strip():
        return {
            "response": "Please enter a message or question.",
            "sources": [],
            "tool_used": "none",
            "intent": "none",
        }

    # Apply input safety
    sanitized_input = apply_safety(user_input)

    # Initialize state
    initial_state: AgentState = {
        "user_input": sanitized_input,
        "intent": "",
        "response": "",
        "sources": [],
        "tool_used": "",
    }

    try:
        agent = get_agent()
        result = agent.invoke(initial_state)

        return {
            "response": result["response"],
            "sources": result.get("sources", []),
            "tool_used": result.get("tool_used", "unknown"),
            "intent": result.get("intent", "unknown"),
        }

    except Exception as e:
        logger.error(f"Agent execution failed: {e}")
        return {
            "response": f"Sorry, I encountered an error processing your request: {str(e)}",
            "sources": [],
            "tool_used": "error",
            "intent": "error",
        }


# --- CLI entry point for testing ---
if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO)

    test_queries = [
        "What does the document say about solar energy?",  # → retrieve
        "What is the current price of Tesla stock?",  # → web_search
        "Calculate 15% of 2500",  # → calculate
        "Hello, what can you do?",  # → direct
        "Tell me about renewable energy in the uploaded PDF",  # → retrieve (ambiguous)
    ]

    if len(sys.argv) > 1:
        # Custom query from command line
        query = " ".join(sys.argv[1:])
        test_queries = [query]

    print("=" * 60)
    print("🤖 AI Document Assistant — Agent Router Test")
    print("=" * 60)

    for query in test_queries:
        print(f"\n{'─' * 60}")
        print(f"❓ Query: {query}")
        result = run_agent(query)
        print(f"🏷️  Intent: {result['intent']}")
        print(f"🔧 Tool: {result['tool_used']}")
        print(f"📝 Response: {result['response'][:200]}...")
        if result["sources"]:
            print(f"📄 Sources: {len(result['sources'])} citations")
