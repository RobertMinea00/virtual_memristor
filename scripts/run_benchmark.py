"""
Run the full benchmark suite.

Usage:
    cd C:/Users/rober/Desktop/Experiment_Mem
    .venv/Scripts/python scripts/run_benchmark.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from benchmarks.runner import BenchmarkRunner

if __name__ == "__main__":
    runner = BenchmarkRunner()
    results = runner.run()
