"""Tests for src/eval.py — RAGAS evaluation module.

Tests cover:
- Q&A pair loading and validation
- Pipeline results collection
- Evaluation scoring (manual fallback)
- Results writing to markdown

All Gemini API calls are mocked — no network or quota burn.
"""

import json
import os
import tempfile

from unittest.mock import patch

from src.eval import (
    load_qa_pairs,
    _manual_evaluation,
    write_results,
)


class TestLoadQaPairs:
    """Tests for the load_qa_pairs function."""

    def test_loads_valid_file(self):
        """Should load Q&A pairs from the project's qa_pairs.json."""
        pairs = load_qa_pairs()
        assert isinstance(pairs, list)
        assert len(pairs) == 10
        assert "question" in pairs[0]
        assert "ground_truth" in pairs[0]

    def test_file_not_found_raises(self):
        """Should raise FileNotFoundError for missing file."""
        try:
            load_qa_pairs("/nonexistent/qa_pairs.json")
            assert False, "Expected FileNotFoundError"
        except FileNotFoundError as e:
            assert "Q&A pairs file not found" in str(e)

    def test_invalid_json_raises(self):
        """Should raise ValueError for malformed JSON."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{invalid json")
            tmp_path = f.name

        try:
            load_qa_pairs(tmp_path)
            assert False, "Expected ValueError"
        except ValueError as e:
            assert "Invalid JSON" in str(e)
        finally:
            os.unlink(tmp_path)

    def test_empty_array_raises(self):
        """Should raise ValueError for empty JSON array."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([], f)
            tmp_path = f.name

        try:
            load_qa_pairs(tmp_path)
            assert False, "Expected ValueError"
        except ValueError as e:
            assert "JSON array" in str(e)
        finally:
            os.unlink(tmp_path)

    def test_missing_keys_raises(self):
        """Should raise ValueError when Q&A pairs lack required keys."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump([{"question": "Q1"}], f)  # Missing ground_truth
            tmp_path = f.name

        try:
            load_qa_pairs(tmp_path)
            assert False, "Expected ValueError"
        except ValueError as e:
            assert "ground_truth" in str(e)
        finally:
            os.unlink(tmp_path)


class TestManualEvaluation:
    """Tests for the _manual_evaluation fallback scorer."""

    def test_returns_three_metrics(self):
        """Should return faithfulness, answer_relevancy, and context_precision."""
        pipeline_results = {
            "questions": ["What is solar energy?"],
            "answers": ["Solar energy comes from the sun."],
            "contexts": [["Solar energy is renewable energy from sunlight."]],
            "ground_truths": ["Solar energy is energy from the sun."],
        }
        scores = _manual_evaluation(pipeline_results)
        assert "faithfulness" in scores
        assert "answer_relevancy" in scores
        assert "context_precision" in scores
        assert all(0 <= v <= 1 for v in scores.values())

    def test_no_context_uncertainty_scored_high(self):
        """Answers admitting uncertainty with no context should get high faithfulness."""
        pipeline_results = {
            "questions": ["What is quantum computing?"],
            "answers": ["I don't have enough information to answer this."],
            "contexts": [[""]],
            "ground_truths": ["Quantum computing uses qubits."],
        }
        scores = _manual_evaluation(pipeline_results)
        assert scores["faithfulness"] == 1.0

    def test_handles_multiple_questions(self):
        """Should average scores across multiple questions."""
        pipeline_results = {
            "questions": ["Q1?", "Q2?"],
            "answers": ["A1 about solar.", "I don't have enough information."],
            "contexts": [["solar panel context"], [""]],
            "ground_truths": ["Solar panels.", "Wind energy."],
        }
        scores = _manual_evaluation(pipeline_results)
        assert all(isinstance(v, float) for v in scores.values())


class TestWriteResults:
    """Tests for the write_results function."""

    def test_writes_markdown_file(self):
        """Should write a valid markdown results file."""
        scores = {
            "faithfulness": 0.85,
            "answer_relevancy": 0.72,
            "context_precision": 0.60,
        }
        pipeline_results = {
            "questions": ["What is solar energy?"],
            "answers": ["Solar energy comes from the sun."],
            "contexts": [["Solar context"]],
            "ground_truths": ["Energy from sunlight."],
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, dir="."
        ) as f:
            tmp_path = f.name

        try:
            write_results(scores, pipeline_results, output_path=tmp_path)

            with open(tmp_path) as f:
                content = f.read()

            assert "# RAG Evaluation Results" in content
            assert "0.8500" in content
            assert "0.7200" in content
            assert "0.6000" in content
            assert "solar energy" in content.lower()
        finally:
            os.unlink(tmp_path)

    @patch.dict("os.environ", {
        "GEMINI_MODEL": "gemini-2.5-flash",
        "GEMINI_EMBEDDING_MODEL": "gemini-embedding-2",
    })
    def test_includes_model_info(self):
        """Should include model names from environment."""
        scores = {"faithfulness": 0.5, "answer_relevancy": 0.5, "context_precision": 0.5}
        pipeline_results = {
            "questions": ["Q?"],
            "answers": ["A."],
            "contexts": [["C"]],
            "ground_truths": ["GT"],
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, dir="."
        ) as f:
            tmp_path = f.name

        try:
            write_results(scores, pipeline_results, output_path=tmp_path)

            with open(tmp_path) as f:
                content = f.read()

            assert "gemini-2.5-flash" in content
            assert "gemini-embedding-2" in content
        finally:
            os.unlink(tmp_path)
