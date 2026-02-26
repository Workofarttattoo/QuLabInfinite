import unittest
from ech0_core.advanced_reasoning import ECH0_Advanced_Reasoning

class TestAdvancedReasoningExtraction(unittest.TestCase):

    def setUp(self):
        self.reasoning = ECH0_Advanced_Reasoning()

    def test_extract_answer_with_marker(self):
        response = "The solution involves integrating over the volume. Therefore, the result is 42. ANSWER: 42"
        extracted = self.reasoning._extract_answer(response)
        self.assertEqual(extracted, "42")

    def test_extract_answer_with_latex(self):
        response = "Here is the calculation. ANSWER: $\\frac{1}{2}$"
        extracted = self.reasoning._extract_answer(response)
        self.assertEqual(extracted, "frac{1}{2}")

    def test_extract_answer_fallback_number(self):
        response = "The final result after calculation is 100.5."
        extracted = self.reasoning._extract_answer(response)
        self.assertEqual(extracted, "100.5")

    def test_extract_answer_fallback_multiple_numbers(self):
        # Should pick the last number
        response = "First part is 10, second is 20. Total is 30."
        extracted = self.reasoning._extract_answer(response)
        self.assertEqual(extracted, "30")

    def test_extract_answer_unknown(self):
        response = "I don't know the answer."
        extracted = self.reasoning._extract_answer(response)
        self.assertEqual(extracted, "Unknown")

    def test_extract_corrected_answer_explicit(self):
        response = "My previous answer was wrong. CORRECTED_ANSWER: 50"
        extracted = self.reasoning._extract_corrected_answer(response)
        self.assertEqual(extracted, "50")

    def test_extract_corrected_answer_implicit(self):
        # Should fallback to _extract_answer behavior
        response = "The recalculation shows the value is actually 75."
        extracted = self.reasoning._extract_corrected_answer(response)
        self.assertEqual(extracted, "75")

if __name__ == '__main__':
    unittest.main()
