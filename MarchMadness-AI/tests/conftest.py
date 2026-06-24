"""Make the ``src/`` layout importable when running the tests in-place.

This lets ``pytest`` work straight from the project root without first
``pip install -e .``-ing the package (though that works too).
"""
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
