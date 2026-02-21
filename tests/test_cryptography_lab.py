
import pytest
from cryptography_lab import CryptographyLab

class TestCryptographyLab:
    @pytest.fixture
    def lab(self):
        return CryptographyLab()

    @pytest.mark.parametrize("n, expected", [
        (-10, False),
        (-1, False),
        (0, False),
        (1, False),
        (2, True),
        (3, True),
        (4, False),
        (5, True),
        (6, False),
        (7, True),
        (8, False),
        (9, False),
        (10, False),
        (11, True),
        (12, False),
        (13, True),
        (14, False),
        (15, False),
        (17, True),
        (19, True),
        (561, False),  # Carmichael number
        (1105, False), # Carmichael number
        (1729, False), # Carmichael number
        (104729, True), # 10000th prime
    ])
    def test_is_prime(self, lab, n, expected):
        assert lab.is_prime(n) == expected

    def test_generate_prime(self, lab):
        for bits in [8, 16, 32]:
            p = lab.generate_prime(bits)
            assert lab.is_prime(p)
            assert p.bit_length() == bits
            assert p % 2 == 1 or p == 2
