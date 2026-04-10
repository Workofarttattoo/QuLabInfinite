import pytest
from unittest.mock import patch, MagicMock
import sys
import builtins

import ech0_prime_qulab_connector
from ech0_prime_qulab_connector import main

@pytest.fixture
def mock_researcher():
    with patch("ech0_prime_qulab_connector.Ech0PrimeQuLabResearcher") as mock_class:
        mock_instance = MagicMock()
        mock_class.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def mock_logging():
    with patch("ech0_prime_qulab_connector.logging.info") as mock_info:
        yield mock_info

def test_main_cli_args(mock_researcher, mock_logging):
    """Test main function when arguments are passed via CLI."""
    mock_researcher.build_session_filepath.return_value = "/path/to/session.json"

    with patch.object(sys, "argv", ["script.py", "test", "question"]):
        main()

    mock_researcher.research_loop.assert_called_once_with("test question", max_iterations=5, verbose=True)
    mock_researcher.save_session.assert_called_once_with("/path/to/session.json")
    mock_logging.assert_any_call("\nSession saved to: /path/to/session.json")

def test_main_interactive_quit(mock_researcher, mock_logging):
    """Test main function interactive loop with 'quit'."""
    with patch.object(sys, "argv", ["script.py"]):
        with patch.object(builtins, "input", side_effect=["quit"]):
            main()

    # If we get here without infinite loop, it worked.

def test_main_interactive_empty_input(mock_researcher, mock_logging):
    """Test main function interactive loop with empty input."""
    with patch.object(sys, "argv", ["script.py"]):
        with patch.object(builtins, "input", side_effect=["", "quit"]):
            main()

    # Continues to prompt if empty input.

def test_main_interactive_labs(mock_researcher, mock_logging):
    """Test main function interactive loop with 'labs'."""
    mock_researcher.qulab.list_labs.return_value = ["quantum_lab", "oncology_lab"]

    with patch.object(sys, "argv", ["script.py"]):
        with patch.object(builtins, "input", side_effect=["labs", "quit"]):
            main()

    mock_researcher.qulab.list_labs.assert_called_once()
    mock_logging.assert_any_call("  - quantum_lab")
    mock_logging.assert_any_call("  - oncology_lab")

def test_main_interactive_labs_empty(mock_researcher, mock_logging):
    """Test main function interactive loop with 'labs' returning empty."""
    mock_researcher.qulab.list_labs.return_value = []

    with patch.object(sys, "argv", ["script.py"]):
        with patch.object(builtins, "input", side_effect=["labs", "quit"]):
            main()

    mock_logging.assert_any_call("\n[INFO] QuLab API not running or no labs returned")

def test_main_interactive_research(mock_researcher, mock_logging):
    """Test main function interactive loop with 'research <q>'."""
    mock_researcher.build_session_filepath.return_value = "session.json"

    with patch.object(sys, "argv", ["script.py"]):
        with patch.object(builtins, "input", side_effect=["research some question", "quit"]):
            main()

    mock_researcher.research_loop.assert_called_once_with("some question", max_iterations=5, verbose=True)
    mock_researcher.save_session.assert_called_once_with("session.json")

def test_main_interactive_single(mock_researcher, mock_logging):
    """Test main function interactive loop with 'single <q>'."""
    mock_researcher.query_llm.return_value = "LLM response"

    with patch.object(sys, "argv", ["script.py"]):
        with patch.object(builtins, "input", side_effect=["single some question", "quit"]):
            main()

    mock_researcher.query_llm.assert_called_once_with("some question")
    mock_logging.assert_any_call("\n[ECH0-PRIME]\nLLM response\n")

def test_main_interactive_other(mock_researcher, mock_logging):
    """Test main function interactive loop with other input."""
    mock_researcher.query_llm.return_value = "LLM response for other"

    with patch.object(sys, "argv", ["script.py"]):
        with patch.object(builtins, "input", side_effect=["hello there", "quit"]):
            main()

    mock_researcher.query_llm.assert_called_once_with("hello there")
    mock_logging.assert_any_call("\n[ECH0-PRIME]\nLLM response for other\n")

def test_main_interactive_eof(mock_researcher, mock_logging):
    """Test main function interactive loop EOFError."""
    with patch.object(sys, "argv", ["script.py"]):
        with patch.object(builtins, "input", side_effect=EOFError):
            main()

    mock_logging.assert_any_call("\nExiting.")

def test_main_interactive_keyboard_interrupt(mock_researcher, mock_logging):
    """Test main function interactive loop KeyboardInterrupt."""
    with patch.object(sys, "argv", ["script.py"]):
        with patch.object(builtins, "input", side_effect=KeyboardInterrupt):
            main()

    mock_logging.assert_any_call("\nExiting.")
