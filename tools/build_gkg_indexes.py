#!/usr/bin/env python3
"""Build gkg_index indexes with live terminal progress."""

import os
import time
import threading
import pymongo
from pymongo import MongoClient

MONGO_URI = os.getenv("LOCAL_MONGO_URI", "mongodb://root:root@127.0.0.1:37018/")
DB_NAME   = "quant_data"
COL_NAME  = "gkg_index"

INDEXES = [
    ("ts",  {"ts": 1},         {"name": "gkg_ts",         "background": True}),
    ("url", {"url": 1},        {"name": "gkg_url_unique", "background": True, "unique": True}),
    ("raw", [("raw", "text")], {"name": "gkg_raw_text",   "background": True}),
]


def get_progress(client):
    try:
        results = list(client["admin"].aggregate([
            {"$currentOp": {"allUsers": True, "idleConnections": False}},
            {"$match": {"msg": {"$regex": "Index Build"}}}
        ]))
        for op in results:
            prog  = op.get("progress", {})
            done  = prog.get("done", 0)
            total = prog.get("total", 1)
            if total == 0:
                continue
            pct     = done / total * 100
            elapsed = op.get("secs_running", {}).get("low", 0)
            msg     = op.get("msg", "")
            return pct, done, total, elapsed, msg
    except Exception:
        pass
    return None


def build_index(col, client, label, keys, opts):
    print(f"\n{'='*60}", flush=True)
    print(f"Building: {label}  {keys}", flush=True)
    print(f"{'='*60}", flush=True)

    result = {"done": False, "error": None}

    def _run():
        try:
            col.create_index(keys, **opts)
        except Exception as e:
            if isinstance(e, pymongo.errors.OperationFailure) and "already exists" in str(e):
                pass  # benign
            else:
                result["error"] = e
        finally:
            result["done"] = True

    t = threading.Thread(target=_run, daemon=True)
    t.start()
    t0 = time.time()

    while not result["done"]:
        info = get_progress(client)
        elapsed = int(time.time() - t0)
        if info:
            pct, done, total, _, key = info
            bar_len = 40
            filled = int(bar_len * pct / 100)
            bar = "█" * filled + "░" * (bar_len - filled)
            done_m  = done  // 1_000_000
            total_m = total // 1_000_000
            eta = int((100 - pct) / (pct / elapsed)) if pct > 0 and elapsed > 0 else 0
            eta_str = f"{eta//3600}h{(eta%3600)//60}m" if eta > 3600 else f"{eta//60}m{eta%60}s"
            print(f"\r[{bar}] {pct:5.1f}%  {done_m}/{total_m}M  elapsed={elapsed//60}m{elapsed%60}s  ETA={eta_str}   ",
                  end="", flush=True)
        else:
            print(f"\r  [{label}] waiting for MongoDB...  elapsed={elapsed}s   ",
                  end="", flush=True)
        time.sleep(3)

    t.join()
    elapsed = int(time.time() - t0)
    if result["error"]:
        print(f"\n  ERROR: {result['error']}", flush=True)
        return False

    # Verify the index actually exists after creation
    existing_after = {v.get("name", k) for k, v in col.index_information().items()}
    if opts["name"] in existing_after:
        print(f"\n  Done in {elapsed//60}m{elapsed%60}s  ✓ verified in index_information", flush=True)
        return True
    else:
        print(f"\n  WARN: create_index returned but '{opts['name']}' not found in index_information — likely rolled back!", flush=True)
        return False


def main():
    print("Connecting to MongoDB...", flush=True)
    client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    client.admin.command("ping")
    col = client[DB_NAME][COL_NAME]

    existing = {v.get("name", k) for k, v in col.index_information().items()}
    print(f"Existing indexes: {existing}", flush=True)

    for label, keys, opts in INDEXES:
        if opts["name"] in existing:
            print(f"Skip {label} — already exists", flush=True)
            continue
        ok = build_index(col, client, label, keys, opts)
        if not ok:
            print(f"  FAILED to build {label}, stopping.", flush=True)
            break

    print("\nAll indexes:", flush=True)
    for idx in col.getIndexes():
        print(f"  {idx['name']}: {idx['key']}", flush=True)


if __name__ == "__main__":
    main()
