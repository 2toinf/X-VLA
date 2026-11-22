import os
import sys

# The repo is not an installed package; make its modules importable so the
# tests run the same way under `pytest` and `python -m pytest`.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
