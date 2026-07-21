import sys
from pathlib import Path

# Make research.<subpackage> modules importable (research/ and its
# subpackages all have __init__.py).
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
