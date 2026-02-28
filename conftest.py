"""
pytest conftest.py — adds the experiment root to sys.path so that all
modules can be imported as absolute packages (e.g. `from llm.client import ...`)
regardless of how pytest is invoked.
"""
import sys
import os

# Insert the experiment root directory at the front of sys.path
_EXPERIMENT_ROOT = os.path.dirname(os.path.abspath(__file__))
if _EXPERIMENT_ROOT not in sys.path:
    sys.path.insert(0, _EXPERIMENT_ROOT)

# Made with Bob
