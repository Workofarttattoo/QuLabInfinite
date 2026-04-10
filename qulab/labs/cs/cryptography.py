"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

CRYPTOGRAPHY LAB
Free gift to the scientific community from QuLabInfinite.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Tuple

# Constants and configuration
DEFAULT_KEY_SIZE = 256


@dataclass
class Cryptography:
    key: np.ndarray = field(default_factory=lambda: np.random.randint(0, 256, DEFAULT_KEY_SIZE, dtype=np.uint8))
    
    @staticmethod
    def generate_key(key_size: int) -> np.ndarray:
        return np.random.randint(0, 256, key_size, dtype=np.uint8)

    def encrypt(self, plaintext: bytes) -> Tuple[np.ndarray, str]:
        if len(plaintext) > self.key.size:
            raise ValueError("Plaintext length exceeds key size")
        
        encrypted_data = (np.frombuffer(plaintext, dtype=np.uint8) ^ self.key)[:len(plaintext)]
        return encrypted_data, base64.b64encode(encrypted_data.tobytes()).decode('utf-8')

    def decrypt(self, ciphertext: str) -> bytes:
        encoded_ciphertext = base64.b64decode(ciphertext)
        decrypted_data = np.frombuffer(encoded_ciphertext, dtype=np.uint8) ^ self.key
        return decrypted_data.tobytes()

def run_demo():
    import base64

    lab = Cryptography()
    
    plaintext = b'Hello, World!'
    encrypted_data, encoded_cipher = lab.encrypt(plaintext)
    print(f"Encrypted Data: {encoded_cipher}")
    
    decoded_plaintext = lab.decrypt(encoded_cipher)
    print(f"Decoded Plaintext: {decoded_plaintext}")

if __name__ == '__main__':
    run_demo()