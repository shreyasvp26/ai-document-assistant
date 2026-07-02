"""RAGAS evaluation module — evaluates the RAG pipeline against ground-truth Q&A pairs.

This module:
1. Loads Q&A pairs from eval/qa_pairs.json
2. Runs each question through the full RAG pipeline
3. Scores results using RAGAS metrics (faithfulness, answer relevancy, context precision)
4. Writes a formatted results table to eval/eval_results.md

Design decisions:
- Uses the RAGAS library with LLM-as-a-judge approach
- Scores are computed per-question and averaged for the final report
- Results are written as markdown for easy inclusion in README
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# --- Configuration ---
QA_PAIRS_PATH = os.path.join(os.path.dirname(__file__), "..", "eval", "qa_pairs.json")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "..", "eval", "eval_results.md")


def load_qa_pairs(path: str = QA_PAIRS_PATH) -> list[dict]:
    """Load question-answer pairs from the JSON file.

    Args:
        path: Path to the qa_pairs.json file.

    Returns:
        List of dicts with 'question' and 'ground_truth' keys.

    Raises:
        FileNotFoundError: If the Q&A pairs file doesn't exist.
        ValueError: If the file is empty or malformed.
    """
    abs_path = os.path.abspath(path)

    if not os.path.exists(abs_path):
        raise FileNotFoundError(
            f"Q&A pairs file not found: {abs_path}. "
            "Please create eval/qa_pairs.json with question/ground_truth pairs."
        )

    try:
        with open(abs_path, "r") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in Q&A pairs file: {e}")

    if not data or not isinstance(data, list):
        raise ValueError(
            "Q&A pairs file must contain a JSON array of objects "
            "with 'question' and 'ground_truth' keys."
        )

    # Validate structure
    for i, pair in enumerate(data):
        if "question" not in pair or "ground_truth" not in pair:
            raise ValueError(
                f"Q&A pair at index {i} missing 'question' or 'ground_truth' key."
            )

    logger.info(f"Loaded {len(data)} Q&A pairs from '{abs_path}'")
    return data


def run_pipeline_on_questions(qa_pairs: list[dict]) -> dict:
    """Run the RAG pipeline on each question and collect results.

    Args:
        qa_pairs: List of dicts with 'question' and 'ground_truth' keys.

    Returns:
        Dict with 'questions', 'answers', 'contexts', 'ground_truths' lists.
    """
    from src.vectorstore import rag_query, query_documents

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for i, pair in enumerate(qa_pairs):
        question = pair["question"]
        ground_truth = pair["ground_truth"]

        logger.info(f"[{i+1}/{len(qa_pairs)}] Processing: '{question[:60]}...'")
        print(f"  [{i+1}/{len(qa_pairs)}] {question[:70]}...")

        try:
            # Get the RAG answer
            result = rag_query(question)
            answer = result.get("answer", "No answer generated")

            # Get the raw retrieved contexts (for RAGAS evaluation)
            raw_contexts = query_documents(question, top_k=5)
            context_texts = [c["text"] for c in raw_contexts] if raw_contexts else [""]

            questions.append(question)
            answers.append(answer)
            contexts.append(context_texts)
            ground_truths.append(ground_truth)

        except Exception as e:
            logger.error(f"Pipeline failed for question '{question}': {e}")
            questions.append(question)
            answers.append(f"Error: {str(e)}")
            contexts.append([""])
            ground_truths.append(ground_truth)

    return {
        "questions": questions,
        "answers": answers,
        "contexts": contexts,
        "ground_truths": ground_truths,
    }


def evaluate_with_ragas(pipeline_results: dict) -> dict:
    """Score the pipeline results using RAGAS metrics.

    Metrics:
    - Faithfulness: Is the answer grounded in the retrieved context?
    - Answer Relevancy: Does the answer address the question?
    - Context Precision: Are the most relevant chunks ranked highest?

    Args:
        pipeline_results: Dict from run_pipeline_on_questions().

    Returns:
        Dict with metric names as keys and scores as values.
    """
    try:
        from datasets import Dataset
        from ragas import evaluate
        from ragas.metrics import faithfulness, answer_relevancy, context_precision

        # Build the RAGAS-compatible dataset
        eval_data = {
            "question": pipeline_results["questions"],
            "answer": pipeline_results["answers"],
            "contexts": pipeline_results["contexts"],
            "ground_truth": pipeline_results["ground_truths"],
        }

        dataset = Dataset.from_dict(eval_data)

        print("\n📊 Running RAGAS evaluation (this may take a few minutes)...")
        logger.info("Starting RAGAS evaluation...")

        results = evaluate(
            dataset,
            metrics=[faithfulness, answer_relevancy, context_precision],
        )

        scores = {
            "faithfulness": float(results.get("faithfulness", 0)),
            "answer_relevancy": float(results.get("answer_relevancy", 0)),
            "context_precision": float(results.get("context_precision", 0)),
        }

        logger.info(f"RAGAS evaluation complete: {scores}")
        return scores

    except ImportError as e:
        logger.error(f"RAGAS import failed: {e}")
        print(f"\n⚠️ RAGAS evaluation requires additional packages: {e}")
        # Provide fallback manual evaluation
        return _manual_evaluation(pipeline_results)

    except Exception as e:
        logger.error(f"RAGAS evaluation failed: {e}")
        print(f"\n⚠️ RAGAS evaluation failed: {e}")
        print("Falling back to manual evaluation...")
        return _manual_evaluation(pipeline_results)


def _manual_evaluation(pipeline_results: dict) -> dict:
    """Fallback evaluation when RAGAS is unavailable.

    Performs basic heuristic scoring:
    - Faithfulness: checks if answer mentions "don't have" for questions with no context
    - Answer relevancy: checks keyword overlap between question and answer
    - Context precision: checks if context contains question keywords

    Args:
        pipeline_results: Dict from run_pipeline_on_questions().

    Returns:
        Dict with metric scores.
    """
    scores = {"faithfulness": [], "answer_relevancy": [], "context_precision": []}

    for i in range(len(pipeline_results["questions"])):
        question = pipeline_results["questions"][i].lower()
        answer = pipeline_results["answers"][i].lower()
        contexts = pipeline_results["contexts"][i]
        ground_truth = pipeline_results["ground_truths"][i].lower()

        # Faithfulness: does the answer contain info from context?
        context_text = " ".join(contexts).lower()
        if context_text.strip():
            # Check keyword overlap between answer and context
            answer_words = set(answer.split())
            context_words = set(context_text.split())
            overlap = len(answer_words & context_words) / max(len(answer_words), 1)
            scores["faithfulness"].append(min(overlap * 2, 1.0))
        else:
            # If no context, faithfulness is 1.0 only if answer admits uncertainty
            if "don't have" in answer or "not enough" in answer:
                scores["faithfulness"].append(1.0)
            else:
                scores["faithfulness"].append(0.3)

        # Answer relevancy: keyword overlap between question and answer
        q_words = set(question.split()) - {"what", "is", "the", "a", "an", "how", "does"}
        a_words = set(answer.split())
        relevancy = len(q_words & a_words) / max(len(q_words), 1)
        scores["answer_relevancy"].append(min(relevancy, 1.0))

        # Context precision: do retrieved contexts contain ground truth keywords?
        gt_words = set(ground_truth.split()) - {"the", "is", "a", "an", "of", "and", "to"}
        if context_text.strip():
            precision = sum(1 for w in gt_words if w in context_text) / max(len(gt_words), 1)
            scores["context_precision"].append(min(precision, 1.0))
        else:
            scores["context_precision"].append(0.0)

    return {
        "faithfulness": sum(scores["faithfulness"]) / max(len(scores["faithfulness"]), 1),
        "answer_relevancy": sum(scores["answer_relevancy"]) / max(len(scores["answer_relevancy"]), 1),
        "context_precision": sum(scores["context_precision"]) / max(len(scores["context_precision"]), 1),
    }


def write_results(
    scores: dict,
    pipeline_results: dict,
    output_path: str = RESULTS_PATH,
) -> None:
    """Write evaluation results to a markdown file.

    Args:
        scores: Dict of metric name → score.
        pipeline_results: The full pipeline results for per-question details.
        output_path: Path to write the results file.
    """
    abs_path = os.path.abspath(output_path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "# RAG Evaluation Results",
        "",
        f"**Evaluated:** {timestamp}",
        f"**Questions:** {len(pipeline_results['questions'])}",
        f"**Model:** {os.getenv('GEMINI_MODEL', 'gemini-2.0-flash')}",
        f"**Embedding:** {os.getenv('GEMINI_EMBEDDING_MODEL', 'gemini-embedding-001')}",
        "",
        "## Overall Scores",
        "",
        "| Metric | Score | Description |",
        "|--------|-------|-------------|",
        f"| **Faithfulness** | {scores.get('faithfulness', 0):.4f} | Answer grounded in retrieved context |",
        f"| **Answer Relevancy** | {scores.get('answer_relevancy', 0):.4f} | Answer addresses the question asked |",
        f"| **Context Precision** | {scores.get('context_precision', 0):.4f} | Most relevant chunks ranked highest |",
        "",
        "## Per-Question Results",
        "",
        "| # | Question | Answer (truncated) |",
        "|---|----------|-------------------|",
    ]

    for i in range(len(pipeline_results["questions"])):
        q = pipeline_results["questions"][i][:60]
        a = pipeline_results["answers"][i][:80].replace("\n", " ")
        lines.append(f"| {i+1} | {q} | {a} |")

    lines.extend([
        "",
        "---",
        "",
        "*Generated by `python -m src.eval`*",
    ])

    with open(abs_path, "w") as f:
        f.write("\n".join(lines))

    logger.info(f"Results written to '{abs_path}'")
    print(f"\n✅ Results written to: {abs_path}")


def run_evaluation() -> dict:
    """Run the complete evaluation pipeline.

    This is the main entry point:
    1. Load Q&A pairs
    2. Run each question through the RAG pipeline
    3. Score with RAGAS
    4. Write results to markdown

    Returns:
        Dict with metric scores.
    """
    print("=" * 60)
    print("📊 AI Document Assistant — RAG Evaluation")
    print("=" * 60)

    # Step 1: Load Q&A pairs
    print("\n📂 Loading Q&A pairs...")
    qa_pairs = load_qa_pairs()
    print(f"   Loaded {len(qa_pairs)} question/answer pairs")

    # Step 2: Run pipeline
    print("\n🔄 Running RAG pipeline on each question...")
    pipeline_results = run_pipeline_on_questions(qa_pairs)

    # Step 3: Evaluate
    print("\n📊 Computing evaluation scores...")
    scores = evaluate_with_ragas(pipeline_results)

    # Step 4: Write results
    print("\n📝 Writing results...")
    write_results(scores, pipeline_results)

    # Print summary
    print("\n" + "=" * 60)
    print("📊 EVALUATION SUMMARY")
    print("=" * 60)
    for metric, score in scores.items():
        emoji = "✅" if score >= 0.7 else "⚠️" if score >= 0.5 else "❌"
        print(f"  {emoji} {metric}: {score:.4f}")
    print("=" * 60)

    return scores


# --- CLI entry point ---
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_evaluation()
