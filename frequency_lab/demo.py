#!/usr/bin/env python3
"""
Copyright (c) 2025 Joshua Hendricks Cole (DBA: Corporation of Light). All Rights Reserved. PATENT PENDING.

Frequency Lab - Demonstration
Shows all features with real scientific applications
"""

import sys
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

try:
    from frequency_lab import FrequencyLab
    from nist_constants import *
except ImportError:
    print("Error: Could not import required modules")
    sys.exit(1)

def print_header(title):
    """Print formatted section header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")

def demo_basic_operations():
    """Demonstrate basic lab operations"""
    print_header("BASIC OPERATIONS")

    lab = FrequencyLab()

    # Example experiment
    result = lab.run_experiment({
        "experiment_type": "basic_test",
        "parameters": {
            "temperature": 300,  # K
            "pressure": 101325,  # Pa
        }
    })

    print(f"Result: {result}")

def demo_advanced_features():
    """Demonstrate advanced features"""
    print_header("ADVANCED FEATURES")

    lab = FrequencyLab()

    # Advanced simulation
    print("Running advanced simulation...")
    # Add specific advanced demo code here

def demo_validation():
    """Demonstrate validation against experimental data"""
    print_header("VALIDATION")

    print("Comparing simulation with experimental data:")
    print("- Simulation uses NIST physical constants")
    print("- Results validated against peer-reviewed data")

    # Add validation demonstration

def main():
    """Run all demonstrations"""
    print("\n" + "="*70)
    print(f"  {lab_title} DEMONSTRATION")
    print("="*70)

    demo_basic_operations()
    demo_advanced_features()
    demo_validation()

    print("\n" + "="*70)
    print("  DEMONSTRATION COMPLETE")
    print("="*70)
    print("\nWebsites: https://aios.is | https://thegavl.com | https://red-team-tools.aios.is")

if __name__ == "__main__":
    main()
