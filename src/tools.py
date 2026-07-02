"""Tool definitions for the AI Document Assistant agent.

Defines three callable tools for the LangGraph agent:
1. retrieve_docs — retrieves relevant document chunks from the vector store
2. web_search — searches the web via Tavily API for current information
3. calculator — safely evaluates arithmetic expressions

Design decisions:
- All tools use LangChain's @tool decorator for automatic function calling integration
- Calculator uses ast.parse with a strict whitelist — NO eval() on raw strings
- Each tool has comprehensive error handling and returns error messages, never crashes
"""

import os
import ast
import math
import logging
import operator

from dotenv import load_dotenv
from langchain_core.tools import tool

from src.vectorstore import rag_query

load_dotenv()
logger = logging.getLogger(__name__)

# --- Calculator: Safe arithmetic evaluation ---

# Whitelist of allowed AST node types for the calculator
# This prevents code injection while allowing basic math
ALLOWED_NODES = {
    ast.Expression,
    ast.BinOp,
    ast.UnaryOp,
    ast.Constant,  # Numbers (int, float)
    # Binary operators
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    # Unary operators
    ast.UAdd,
    ast.USub,
}

# Supported binary operators mapped to their implementations
OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

# Supported unary operators
UNARY_OPERATORS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_eval_node(node: ast.AST) -> float:
    """Recursively evaluate an AST node using only whitelisted operations.

    This is the core of the safe calculator — it walks the AST tree
    and evaluates each node using explicit operator functions, never
    Python's built-in eval().

    Args:
        node: An AST node to evaluate.

    Returns:
        The numeric result of evaluating the node.

    Raises:
        ValueError: If an unsupported node type is encountered.
        ZeroDivisionError: If division by zero is attempted.
    """
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"Unsupported constant type: {type(node.value).__name__}")

    elif isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in OPERATORS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")

        left = _safe_eval_node(node.left)
        right = _safe_eval_node(node.right)

        # Guard against excessively large exponents (DoS prevention)
        if op_type == ast.Pow and abs(right) > 1000:
            raise ValueError("Exponent too large (max 1000). This is a safety limit.")

        return OPERATORS[op_type](left, right)

    elif isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in UNARY_OPERATORS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        return UNARY_OPERATORS[op_type](_safe_eval_node(node.operand))

    else:
        raise ValueError(
            f"Unsupported expression element: {type(node).__name__}. "
            "Only basic arithmetic is supported (+, -, *, /, //, %, **)."
        )


def safe_calculate(expression: str) -> str:
    """Safely evaluate a mathematical expression without using eval().

    Parses the expression into an AST and evaluates it using only
    whitelisted arithmetic operations. This prevents code injection
    while supporting basic math.

    Args:
        expression: A mathematical expression string (e.g., "2 + 3 * 4").

    Returns:
        String representation of the result, or an error message.

    Examples:
        >>> safe_calculate("2 + 3 * 4")
        '14.0'
        >>> safe_calculate("(100 - 32) * 5 / 9")
        '37.77777777777778'
        >>> safe_calculate("import os")
        'Error: ...'
    """
    if not expression or not expression.strip():
        return "Error: Empty expression. Please provide a mathematical expression."

    # Clean the expression
    expr = expression.strip()

    try:
        # Parse into AST — this catches syntax errors
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as e:
        return (
            f"Error: Invalid mathematical expression: '{expr}'. "
            f"Syntax error: {str(e)}. "
            "Please use standard math notation (e.g., '2 + 3 * 4')."
        )

    # Validate all nodes are in the whitelist
    for node in ast.walk(tree):
        if type(node) not in ALLOWED_NODES:
            return (
                f"Error: Unsupported operation in expression: '{expr}'. "
                f"Found '{type(node).__name__}' which is not allowed. "
                "Only basic arithmetic is supported (+, -, *, /, //, %, **)."
            )

    # Evaluate the validated AST
    try:
        result = _safe_eval_node(tree.body)
        # Format nicely — show integers without decimal if possible
        if result == int(result):
            return str(int(result))
        return str(result)
    except ZeroDivisionError:
        return "Error: Division by zero is not allowed."
    except ValueError as e:
        return f"Error: {str(e)}"
    except OverflowError:
        return "Error: Result is too large to compute."


# --- Tool Definitions ---


@tool
def retrieve_docs(query: str) -> str:
    """Search uploaded documents to find relevant information.

    Use this tool when the user asks a question about the content of their
    uploaded PDF documents. Returns an answer with source citations.

    Args:
        query: The question to search the documents for.

    Returns:
        An answer based on the document content, with source citations,
        or an error message if something goes wrong.
    """
    try:
        result = rag_query(query)

        if "error" in result:
            return f"Error searching documents: {result['error']}"

        # Format the response with sources
        answer = result["answer"]
        sources = result.get("sources", [])

        if sources:
            source_text = "\n\nSources:\n"
            seen = set()
            for s in sources:
                key = f"{s['document']}-p{s['page']}"
                if key not in seen:
                    source_text += f"- {s['document']}, Page {s['page']}\n"
                    seen.add(key)
            answer += source_text

        return answer

    except Exception as e:
        logger.error(f"retrieve_docs tool failed: {e}")
        return f"Error: Failed to search documents — {str(e)}"


@tool
def web_search(query: str) -> str:
    """Search the web for current information using Tavily.

    Use this tool when the user asks about recent events, current data,
    or information that would not be found in their uploaded documents.

    Args:
        query: The search query.

    Returns:
        Search results summary, or an error message if the search fails.
    """
    tavily_key = os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        return (
            "Error: Web search is not available — TAVILY_API_KEY is not configured. "
            "Get a free API key at https://tavily.com and add it to your .env file."
        )

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=tavily_key)
        response = client.search(query, max_results=3)

        if not response.get("results"):
            return f"No web results found for: '{query}'"

        # Format results into a readable summary
        results_text = f"Web search results for '{query}':\n\n"
        for i, result in enumerate(response["results"], 1):
            title = result.get("title", "No title")
            content = result.get("content", "No content")
            url = result.get("url", "")
            results_text += f"{i}. **{title}**\n"
            results_text += f"   {content[:300]}...\n"
            results_text += f"   Source: {url}\n\n"

        return results_text

    except ImportError:
        return "Error: tavily-python package is not installed. Run: pip install tavily-python"
    except ConnectionError as e:
        logger.error(f"Web search network error: {e}")
        return "Error: Web search failed — check your internet connection."
    except Exception as e:
        logger.error(f"Web search failed: {e}")
        return f"Error: Web search failed — {str(e)}"


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely.

    Use this tool when the user asks to calculate something or needs
    arithmetic computation. Supports +, -, *, /, //, %, and ** (power).

    Args:
        expression: A mathematical expression (e.g., "2 + 3 * 4", "(100 - 32) * 5 / 9").

    Returns:
        The result of the calculation, or an error message for invalid expressions.
    """
    logger.info(f"Calculator called with expression: '{expression}'")
    return safe_calculate(expression)
