#!/usr/bin/env python3
"""Build gkg_index indexes one by one, with live progress polling."""

import os
import time
import threading
import pymongo
from pymongo import MongoClient

MONGO_URI = os.getenv("LOCAL_MONGO_URI", "mongodb://root:root@127.0.0.1:37018/")
DB_NAME   = "quant_data"
COL_NAME  = "gkg_index"

INDEXES = [
    ("ts",  {"ts": 1},          {"name": "gkg_ts",          "background": True}),
    ("url", {"url": 1},         {"name": "gkg_url_unique",  "background": True, "unique": True}),
    ("raw", [("raw", "text")],  {"name": "gkg_raw_text",    "background": True}),
]


def poll_progress(client, stop_event, label):
    db = client["quant_data"]
    t0 = time.time()
    while not stop_event.is_set():
        try:
            ops = db.command("currentOp")
            found = False
            for op in ops.get("inprog", []):
                msg = op.get("msg", "")
                desc = str(op)
                if "index" in msg.lower() or "Index" in desc:
                    pct = ""
                    prog = op.get("progress", {})
                    if prog and prog.get("total", 0) > 0:
                        pct = f"  {prog['done']/prog['total']*100:.1f}%  ({prog['done']:,}/{prog['total']:,})"
                    elapsed = int(time.time() - t0)
                    print(f"  [{label}] {msg or 'building...'}{pct}  elapsed={elapsed}s", flush=True)
                    found = True
                    break
            if not found:
                elapsed = int(time.time() - t0)
                print(f"  [{label}] building...  elapsed={elapsed}s", flush=True)
        except Exception:
            pass
        time.sleep(5)


def build_index(col, client, label, keys, opts):
    print(f"\n{'='*60}")
    print(f"Building index: {label}  keys={keys}")
    print(f"{'='*60}")
    stop = threading.Event()
    t = threading.Thread(target=poll_progress, args=(client, stop, label), daemon=True)
    t.start()
    t0 = time.time()
    try:
        col.create_index(keys, **opts)
        elapsed = time.time() - t0
        print(f"\n  Done in {elapsed/60:.1f} min")
    except pymongo.errors.OperationFailure as e:
        if "already exists" in str(e):
            print(f"\n  Already exists — skipped")
        else:
            raise
    finally:
        stop.set()


def main():
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    col = client[DB_NAME][COL_NAME]

    existing = {v.get("name", k) for k, v in col.index_information().items()}
    print(f"Existing indexes: {existing}")

    for label, keys, opts in INDEXES:
        if opts["name"] in existing:
            print(f"\nSkip {label} — already exists")
            continue
        build_index(col, client, label, keys, opts)

    print("\nAll indexes built.")
    for name, info in col.index_information().items():
        print(f"  {name}: {info['key']}")


if __name__ == "__main__":
    main()
