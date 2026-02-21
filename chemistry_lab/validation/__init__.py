"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

__init__ - Part of Chemistry Lab
"""

"""Validation utilities for Chemistry Lab."""

from .kinetics_validation import KINETICS_BENCHMARKS, run_kinetics_validation

__all__ = [
    "KINETICS_BENCHMARKS",
    "run_kinetics_validation",
]
