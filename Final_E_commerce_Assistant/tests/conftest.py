# tests/conftest.py
import os
import sys

# Add the project root to PYTHONPATH so "backend" can be imported
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)
