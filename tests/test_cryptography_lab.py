
import pytest
import hashlib
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

    def test_merkle_tree_root_empty(self, lab):
        assert lab.merkle_tree_root([]) == b''

    def test_merkle_tree_root_single(self, lab):
        leaf = b'single_leaf'
        assert lab.merkle_tree_root([leaf]) == leaf

    def test_merkle_tree_root_even(self, lab):
        leaves = [b'leaf1', b'leaf2']
        expected = hashlib.sha256(b'leaf1' + b'leaf2').digest()
        assert lab.merkle_tree_root(leaves) == expected

    def test_merkle_tree_root_odd(self, lab):
        leaves = [b'leaf1', b'leaf2', b'leaf3']
        # Implementation pads [l1, l2, l3] -> [l1, l2, l3, l3]
        h12 = hashlib.sha256(b'leaf1' + b'leaf2').digest()
        h33 = hashlib.sha256(b'leaf3' + b'leaf3').digest()
        expected = hashlib.sha256(h12 + h33).digest()
        assert lab.merkle_tree_root(leaves) == expected

    def test_merkle_tree_root_no_side_effect(self, lab):
        leaves = [b'leaf1', b'leaf2', b'leaf3']
        original_leaves = leaves.copy()
        lab.merkle_tree_root(leaves)
        assert leaves == original_leaves
