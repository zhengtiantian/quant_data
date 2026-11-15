import sys
from pathlib import Path

# Make research.<subpackage> modules importable (research/ and its
# subpackages all have __init__.py).
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


class _Cursor(list):
    def sort(self, key, direction=1):
        return _Cursor(sorted(self, key=lambda d: _get(d, key) or "", reverse=direction < 0))

    def limit(self, n):
        return _Cursor(self[:n])


def _get(d, path):
    for k in path.split("."):
        d = d.get(k) if isinstance(d, dict) else None
    return d


def _match(v, cond):
    if not isinstance(cond, dict):
        return v == cond or (isinstance(v, list) and cond in v)
    for op, x in cond.items():
        if op == "$gte" and not (v is not None and v >= x): return False
        if op == "$lt" and not (v is not None and v < x): return False
        if op == "$ne" and v == x: return False
        if op == "$in" and not (any(e in x for e in v) if isinstance(v, list) else v in x): return False
        if op == "$nin" and v in x: return False
    return True


class FakeCollection:
    """The few pymongo find() features the N.6 loaders use ($gte $lt $ne $in $nin on dotted paths, sort, limit);
    projections are ignored."""
    def __init__(self, docs):
        self.docs = list(docs)

    def find(self, q=None, proj=None):
        return _Cursor(d for d in self.docs if all(_match(_get(d, k), c) for k, c in (q or {}).items()))
