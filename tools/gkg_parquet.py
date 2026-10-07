"""GKG index as month-partitioned Parquet, searched with DuckDB.

MongoDB `quant_data.gkg_index` stays the source of truth and keeps only the small
`url` / `ts` indexes it needs for de-duplication during ingestion. Full-history
keyword search and statistics run here instead: the 325GB `$text` index that used
to serve them made every GKG import take 8-17s, then turned out corrupt (checksum
error, 2026-10-07) and was dropped. Without it imports run ~20x faster.

Layout:  <GKG_PARQUET_DIR>/month=YYYY-MM/part-0.parquet   columns: ts, url, raw
Each month file is rebuilt whole from MongoDB, so re-exporting a month is idempotent.

Usage:
  python tools/gkg_parquet.py export --all                 # every month in gkg_index
  python tools/gkg_parquet.py export --months 2026-08 2026-09
  python tools/gkg_parquet.py export --since-last          # last exported month onward (incremental)
  python tools/gkg_parquet.py search "Palantir" --from 2020-01 --to 2026-10 --limit 20
  python tools/gkg_parquet.py search "Palantir" "PLTR" --count
  python tools/gkg_parquet.py stats
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

PARQUET_DIR = Path(os.getenv("GKG_PARQUET_DIR", "/Volumes/Data4T/gkg_parquet"))
STATE_FILE = PARQUET_DIR / "_export_state.json"
MONGO_URI = os.getenv("LOCAL_MONGO_URI") or os.getenv("MONGO_URI") or "mongodb://root:root@127.0.0.1:37018/"
DB_NAME = os.getenv("FEATURE_DB_NAME", "quant_data")
COLLECTION = os.getenv("GKG_COLLECTION", "gkg_index")
CHUNK_ROWS = int(os.getenv("GKG_EXPORT_CHUNK_ROWS", "200000"))

SCHEMA = pa.schema([("ts", pa.string()), ("url", pa.string()), ("raw", pa.string())])


# ---------------------------------------------------------------- export

def _collection():
    from pymongo import MongoClient
    return MongoClient(MONGO_URI, serverSelectionTimeoutMS=20000, socketTimeoutMS=600000)[DB_NAME][COLLECTION]


def _next_month(month: str) -> str:
    y, m = map(int, month.split("-"))
    return f"{y + (m == 12):04d}-{m % 12 + 1:02d}"


def _months_in_mongo(col) -> list[str]:
    """First and last month present (via the ts index), and every month in between."""
    first = col.find_one({}, {"ts": 1, "_id": 0}, sort=[("ts", 1)])
    last = col.find_one({}, {"ts": 1, "_id": 0}, sort=[("ts", -1)])
    if not first or not last:
        return []
    m, end, out = f"{first['ts'][:4]}-{first['ts'][4:6]}", f"{last['ts'][:4]}-{last['ts'][4:6]}", []
    while m <= end:
        out.append(m)
        m = _next_month(m)
    return out


def export_month(col, month: str) -> int:
    """Rebuild month=<month>/part-0.parquet from MongoDB. Returns rows written."""
    lo, hi = month.replace("-", ""), _next_month(month).replace("-", "")
    out_dir = PARQUET_DIR / f"month={month}"
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp, final = out_dir / "part-0.parquet.tmp", out_dir / "part-0.parquet"
    cursor = col.find({"ts": {"$gte": lo, "$lt": hi}}, {"_id": 0, "ts": 1, "url": 1, "raw": 1},
                      batch_size=20000).hint("gkg_ts").sort("ts", 1)
    rows, buf = 0, {"ts": [], "url": [], "raw": []}
    with pq.ParquetWriter(tmp, SCHEMA, compression="zstd") as writer:
        for d in cursor:
            buf["ts"].append(d.get("ts", ""))
            buf["url"].append(d.get("url", ""))
            buf["raw"].append(d.get("raw", ""))
            if len(buf["ts"]) >= CHUNK_ROWS:
                writer.write_table(pa.table(buf, schema=SCHEMA))
                rows += len(buf["ts"])
                buf = {"ts": [], "url": [], "raw": []}
        if buf["ts"]:
            writer.write_table(pa.table(buf, schema=SCHEMA))
            rows += len(buf["ts"])
    if rows:
        tmp.replace(final)
    else:
        tmp.unlink(missing_ok=True)
    return rows


def _load_state() -> dict:
    try:
        return json.loads(STATE_FILE.read_text())
    except Exception:
        return {"months": {}}


def _save_state(state: dict) -> None:
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    STATE_FILE.write_text(json.dumps(state, indent=1, sort_keys=True))


def cmd_export(args) -> None:
    col = _collection()
    state = _load_state()
    available = _months_in_mongo(col)
    if args.all:
        months = available
    elif args.since_last:
        done = sorted(state["months"])
        start = done[-1] if done else (available[0] if available else None)
        months = [m for m in available if start and m >= start]
    else:
        months = args.months or []
    if not months:
        print("Nothing to export.")
        return
    print(f"Exporting {len(months)} month(s) to {PARQUET_DIR}: {months[0]} … {months[-1]}")
    total, t_all = 0, time.time()
    for m in months:
        t0 = time.time()
        n = export_month(col, m)
        total += n
        if n:
            state["months"][m] = {"rows": n, "exported_at": datetime.now(timezone.utc).isoformat(timespec="seconds")}
            _save_state(state)
        print(f"  {m}: {n:,} rows in {time.time() - t0:.1f}s", flush=True)
    print(f"Done: {total:,} rows in {time.time() - t_all:.0f}s")


# ---------------------------------------------------------------- query

def _glob() -> str:
    return str(PARQUET_DIR / "month=*" / "*.parquet")


def search(terms: list[str], month_from: str | None = None, month_to: str | None = None,
           limit: int | None = 50, count_only: bool = False):
    """Rows whose raw text contains any of `terms` (case-insensitive), newest first."""
    import duckdb
    where, params = [], []
    if month_from:
        where.append("ts >= ?"); params.append(month_from.replace("-", ""))
    if month_to:
        where.append("ts < ?"); params.append(_next_month(month_to).replace("-", ""))
    where.append("(" + " OR ".join(["raw ILIKE ?"] * len(terms)) + ")")
    params += [f"%{t}%" for t in terms]
    src = f"read_parquet('{_glob()}', hive_partitioning = true)"
    con = duckdb.connect()
    if count_only:
        return con.execute(f"SELECT count(*) FROM {src} WHERE {' AND '.join(where)}", params).fetchone()[0]
    sql = f"SELECT ts, url, raw FROM {src} WHERE {' AND '.join(where)} ORDER BY ts DESC"
    if limit:
        sql += f" LIMIT {int(limit)}"
    return con.execute(sql, params).fetchall()


def cmd_search(args) -> None:
    t0 = time.time()
    res = search(args.terms, args.month_from, args.month_to, args.limit, args.count)
    if args.count:
        print(f"{res:,} matching rows ({time.time() - t0:.1f}s)")
        return
    for ts, url, raw in res:
        print(f"{ts}  {raw[:110]}\n                {url}")
    print(f"{len(res)} rows ({time.time() - t0:.1f}s)")


def cmd_stats(_args) -> None:
    import duckdb
    t0 = time.time()
    con = duckdb.connect()
    rows = con.execute(f"""SELECT month, count(*) FROM read_parquet('{_glob()}', hive_partitioning = true)
                           GROUP BY month ORDER BY month""").fetchall()
    size = sum(p.stat().st_size for p in PARQUET_DIR.glob("month=*/*.parquet"))
    for m, n in rows:
        print(f"  {m}: {n:,}")
    print(f"{len(rows)} months, {sum(n for _, n in rows):,} rows, {size / 1e9:.1f} GB on disk ({time.time() - t0:.1f}s)")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    g = e.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true")
    g.add_argument("--since-last", action="store_true")
    g.add_argument("--months", nargs="+")
    s = sub.add_parser("search")
    s.add_argument("terms", nargs="+")
    s.add_argument("--from", dest="month_from")
    s.add_argument("--to", dest="month_to")
    s.add_argument("--limit", type=int, default=20)
    s.add_argument("--count", action="store_true")
    sub.add_parser("stats")
    args = ap.parse_args(argv)
    {"export": cmd_export, "search": cmd_search, "stats": cmd_stats}[args.cmd](args)


if __name__ == "__main__":
    main(sys.argv[1:])
