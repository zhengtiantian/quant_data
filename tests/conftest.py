import sys
from pathlib import Path

# Make research/ modules importable (no package __init__ there).
RESEARCH = Path(__file__).resolve().parents[1] / "research"
sys.path.insert(0, str(RESEARCH))
