import unittest
import math
from qulab_ai.tools import calc

class TestCalcSecurity(unittest.TestCase):

    def test_basic_arithmetic(self):
        self.assertEqual(calc("2 + 2"), 4)
        self.assertEqual(calc("10 - 3"), 7)
        self.assertEqual(calc("3 * 4"), 12)
        self.assertEqual(calc("20 / 4"), 5)
        self.assertEqual(calc("2 ** 3"), 8)
        self.assertEqual(calc("4 + 2 * 3"), 10)
        self.assertEqual(calc("(4 + 2) * 3"), 18)
        self.assertEqual(calc("-5 + 3"), -2)
        self.assertEqual(calc("+5 - 2"), 3)

    def test_math_functions(self):
        self.assertAlmostEqual(calc("sqrt(16)"), 4.0)
        self.assertAlmostEqual(calc("sin(0)"), 0.0)
        self.assertAlmostEqual(calc("cos(0)"), 1.0)
        self.assertAlmostEqual(calc("tan(0)"), 0.0)
        self.assertAlmostEqual(calc("abs(-5)"), 5)
        self.assertEqual(calc("round(3.14159)"), 3)
        self.assertEqual(calc("ceil(3.1)"), 4)
        self.assertEqual(calc("floor(3.9)"), 3)
        self.assertAlmostEqual(calc("exp(0)"), 1.0)
        self.assertAlmostEqual(calc("log(e)"), 1.0)

    def test_constants(self):
        self.assertAlmostEqual(calc("pi"), math.pi)
        self.assertAlmostEqual(calc("e"), math.e)

    def test_syntax_errors(self):
        with self.assertRaises(ValueError):
            calc("2 +")
        with self.assertRaises(ValueError):
            calc("2 * * 3")
        with self.assertRaises(ValueError):
            calc("sqrt(16")

    def test_unknown_functions(self):
        with self.assertRaises(ValueError):
            calc("unknown_func(10)")
        with self.assertRaises(ValueError):
            calc("print('hello')")

    def test_unknown_variables(self):
        with self.assertRaises(ValueError):
            calc("x + y")
        with self.assertRaises(ValueError):
            calc("sys.version")

    def test_security_imports(self):
        with self.assertRaises(ValueError):
            calc("__import__('os').system('ls')")
        with self.assertRaises(ValueError):
            calc("import os")
        with self.assertRaises(ValueError):
            calc("from math import sqrt")

    def test_security_attributes(self):
        with self.assertRaises(ValueError):
            calc("'hello'.upper()")
        with self.assertRaises(ValueError):
            calc("(1).__class__")

    def test_security_long_input(self):
        long_expr = "1 + " * 300 + "1"
        with self.assertRaises(ValueError) as context:
            calc(long_expr)
        self.assertIn("Expression too long", str(context.exception))

    def test_complex_expressions(self):
        expr = "sin(pi/2) + cos(0) * sqrt(4)"
        # 1 + 1 * 2 = 3
        self.assertAlmostEqual(calc(expr), 3.0)

    def test_invalid_types(self):
        with self.assertRaises(ValueError):
            calc("'string'")
        with self.assertRaises(ValueError):
            calc("[1, 2, 3]")
        with self.assertRaises(ValueError):
            calc("{'a': 1}")

if __name__ == '__main__':
    unittest.main()
