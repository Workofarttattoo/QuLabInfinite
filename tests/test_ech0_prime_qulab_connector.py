import sys
import unittest
from unittest.mock import patch, MagicMock
from pathlib import Path

# Mock external dependencies that might not be installed
sys.modules['together'] = MagicMock()
sys.modules['requests'] = MagicMock()

# Import the module to test
import ech0_prime_qulab_connector

class TestEch0PrimeQuLabConnectorMain(unittest.TestCase):

    @patch('ech0_prime_qulab_connector.Ech0PrimeQuLabResearcher')
    @patch('ech0_prime_qulab_connector.logging')
    @patch('sys.argv', ['ech0_prime_qulab_connector.py', 'test', 'question'])
    def test_main_with_args(self, mock_logging, mock_researcher_class):
        # Setup mock
        mock_researcher = MagicMock()
        mock_researcher_class.return_value = mock_researcher
        mock_researcher.build_session_filepath.return_value = Path("test/path.json")

        # Call main
        ech0_prime_qulab_connector.main()

        # Assertions
        mock_researcher_class.assert_called_once()
        mock_researcher.research_loop.assert_called_once_with('test question', max_iterations=5, verbose=True)
        mock_researcher.build_session_filepath.assert_called_once()
        mock_researcher.save_session.assert_called_once_with(Path("test/path.json"))

    @patch('ech0_prime_qulab_connector.Ech0PrimeQuLabResearcher')
    @patch('ech0_prime_qulab_connector.logging')
    @patch('builtins.input', side_effect=['labs', 'quit'])
    @patch('sys.argv', ['ech0_prime_qulab_connector.py'])
    def test_main_interactive_labs(self, mock_input, mock_logging, mock_researcher_class):
        # Setup mock
        mock_researcher = MagicMock()
        mock_researcher_class.return_value = mock_researcher
        mock_researcher.qulab.list_labs.return_value = ['lab1', 'lab2']

        # Call main
        ech0_prime_qulab_connector.main()

        # Assertions
        mock_researcher.qulab.list_labs.assert_called_once()
        # Verify logging called with expected lab names
        mock_logging.info.assert_any_call("  - lab1")
        mock_logging.info.assert_any_call("  - lab2")

    @patch('ech0_prime_qulab_connector.Ech0PrimeQuLabResearcher')
    @patch('ech0_prime_qulab_connector.logging')
    @patch('builtins.input', side_effect=['labs', 'quit'])
    @patch('sys.argv', ['ech0_prime_qulab_connector.py'])
    def test_main_interactive_labs_empty(self, mock_input, mock_logging, mock_researcher_class):
        # Setup mock
        mock_researcher = MagicMock()
        mock_researcher_class.return_value = mock_researcher
        mock_researcher.qulab.list_labs.return_value = []

        # Call main
        ech0_prime_qulab_connector.main()

        # Assertions
        mock_researcher.qulab.list_labs.assert_called_once()
        mock_logging.info.assert_any_call("\n[INFO] QuLab API not running or no labs returned")

    @patch('ech0_prime_qulab_connector.Ech0PrimeQuLabResearcher')
    @patch('ech0_prime_qulab_connector.logging')
    @patch('builtins.input', side_effect=['research my topic', 'quit'])
    @patch('sys.argv', ['ech0_prime_qulab_connector.py'])
    def test_main_interactive_research(self, mock_input, mock_logging, mock_researcher_class):
        # Setup mock
        mock_researcher = MagicMock()
        mock_researcher_class.return_value = mock_researcher
        mock_researcher.build_session_filepath.return_value = Path("test/path.json")

        # Call main
        ech0_prime_qulab_connector.main()

        # Assertions
        mock_researcher.research_loop.assert_called_once_with('my topic', max_iterations=5, verbose=True)
        mock_researcher.build_session_filepath.assert_called_once()
        mock_researcher.save_session.assert_called_once_with(Path("test/path.json"))

    @patch('ech0_prime_qulab_connector.Ech0PrimeQuLabResearcher')
    @patch('ech0_prime_qulab_connector.logging')
    @patch('builtins.input', side_effect=['single one question', 'quit'])
    @patch('sys.argv', ['ech0_prime_qulab_connector.py'])
    def test_main_interactive_single(self, mock_input, mock_logging, mock_researcher_class):
        # Setup mock
        mock_researcher = MagicMock()
        mock_researcher_class.return_value = mock_researcher
        mock_researcher.query_llm.return_value = "Test response"

        # Call main
        ech0_prime_qulab_connector.main()

        # Assertions
        mock_researcher.query_llm.assert_called_once_with('one question')
        mock_logging.info.assert_any_call("\n[ECH0-PRIME]\nTest response\n")

    @patch('ech0_prime_qulab_connector.Ech0PrimeQuLabResearcher')
    @patch('ech0_prime_qulab_connector.logging')
    @patch('builtins.input', side_effect=['general question', 'quit'])
    @patch('sys.argv', ['ech0_prime_qulab_connector.py'])
    def test_main_interactive_general(self, mock_input, mock_logging, mock_researcher_class):
        # Setup mock
        mock_researcher = MagicMock()
        mock_researcher_class.return_value = mock_researcher
        mock_researcher.query_llm.return_value = "General response"

        # Call main
        ech0_prime_qulab_connector.main()

        # Assertions
        mock_researcher.query_llm.assert_called_once_with('general question')
        mock_logging.info.assert_any_call("\n[ECH0-PRIME]\nGeneral response\n")

    @patch('ech0_prime_qulab_connector.Ech0PrimeQuLabResearcher')
    @patch('ech0_prime_qulab_connector.logging')
    @patch('builtins.input', side_effect=EOFError)
    @patch('sys.argv', ['ech0_prime_qulab_connector.py'])
    def test_main_interactive_eof(self, mock_input, mock_logging, mock_researcher_class):
        # Call main
        ech0_prime_qulab_connector.main()

        # Assertions
        mock_logging.info.assert_any_call("\nExiting.")

    @patch('ech0_prime_qulab_connector.Ech0PrimeQuLabResearcher')
    @patch('ech0_prime_qulab_connector.logging')
    @patch('builtins.input', side_effect=KeyboardInterrupt)
    @patch('sys.argv', ['ech0_prime_qulab_connector.py'])
    def test_main_interactive_keyboard_interrupt(self, mock_input, mock_logging, mock_researcher_class):
        # Call main
        ech0_prime_qulab_connector.main()

        # Assertions
        mock_logging.info.assert_any_call("\nExiting.")

    @patch('ech0_prime_qulab_connector.Ech0PrimeQuLabResearcher')
    @patch('ech0_prime_qulab_connector.logging')
    @patch('builtins.input', side_effect=['', 'quit'])
    @patch('sys.argv', ['ech0_prime_qulab_connector.py'])
    def test_main_interactive_empty_input(self, mock_input, mock_logging, mock_researcher_class):
        # Call main
        ech0_prime_qulab_connector.main()

        # Should loop over empty string and hit quit
        self.assertEqual(mock_input.call_count, 2)

if __name__ == '__main__':
    unittest.main()
