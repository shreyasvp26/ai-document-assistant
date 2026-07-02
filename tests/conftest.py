"""Shared test fixtures for the AI Document Assistant test suite.

Provides reusable fixtures for sample data, mocked LLM responses,
and temporary file paths to keep individual test files focused.
"""

import os
import pytest
from unittest.mock import MagicMock


@pytest.fixture
def sample_pdf_path():
    """Return the path to the sample PDF in the data/ directory."""
    path = os.path.join(os.path.dirname(__file__), "..", "data", "sample.pdf")
    return os.path.abspath(path)


@pytest.fixture
def sample_text_chunks():
    """Return a list of pre-built text chunks simulating PDF extraction.

    These chunks mirror what ingestion.py would produce from sample.pdf,
    allowing tests to run without actual PDF processing.
    """
    return [
        {
            "text": (
                "Solar photovoltaic (PV) technology converts sunlight directly into "
                "electricity using semiconductor materials. The most common material "
                "is crystalline silicon, which accounts for over 90% of the global "
                "PV market. When photons strike the silicon cell, they knock electrons "
                "free, creating an electric current."
            ),
            "metadata": {"source": "sample.pdf", "page": 1},
        },
        {
            "text": (
                "Wind energy harnesses the kinetic energy of moving air to generate "
                "electricity. Modern wind turbines can have rotor diameters exceeding "
                "150 meters and generate up to 15 MW of power. Offshore wind farms "
                "benefit from stronger, more consistent wind patterns compared to "
                "onshore installations."
            ),
            "metadata": {"source": "sample.pdf", "page": 2},
        },
        {
            "text": (
                "Energy storage systems are critical for managing the intermittent "
                "nature of renewable energy sources. Lithium-ion batteries currently "
                "dominate the market, but emerging technologies like solid-state "
                "batteries and flow batteries offer promising alternatives. Grid-scale "
                "storage capacity reached 45 GW globally in 2024."
            ),
            "metadata": {"source": "sample.pdf", "page": 3},
        },
    ]


@pytest.fixture
def mock_llm_response():
    """Return a mock LLM response object matching Gemini's response structure."""
    mock = MagicMock()
    mock.content = "This is a mocked LLM response for testing purposes."
    return mock


@pytest.fixture
def mock_embedding():
    """Return a mock embedding vector (768-dimensional, matching Gemini embedding output)."""
    return [0.01] * 768
