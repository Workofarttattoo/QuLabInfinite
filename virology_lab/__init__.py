# Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

"""
Virology Laboratory Module
Advanced viral replication modeling, mutation tracking, antiviral drug efficacy, pandemic spread simulation
"""

from .virology_engine import VirologyEngine

__all__ = ['VirologyEngine', 'run_demo']


def run_demo():
    """Lazy import to avoid circular references during package import."""
    from .demo import main as demo_main
    demo_main()
