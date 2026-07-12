#!/usr/bin/env bash
# Build indexes inside the container via mongosh; poll progress from the Mac side.
# Usage: caffeinate -i -s bash tools/build_gkg_indexes_mongosh.sh
#
# Ctrl+C only stops the progress display — it does not interrupt the index build inside the container.
# Re-run the script to resume progress monitoring.

set -euo pipefail

MONGO_INNER="mongodb://root:root@127.0.0.1:27017/?authSource=admin"
MONGO_OUTER="mongodb://root:root@127.0.0.1:37018/"
VENV=".venv311/bin/python"
LOG_URL="/tmp/build_url.log"
LOG_TEXT="/tmp/build_text.log"

# Wait for an index to appear in index_information
wait_for_index() {
    local idx_name="$1"
    while true; do
        exists=$($VENV - <<PYEOF
from pymongo import MongoClient
c = MongoClient("$MONGO_OUTER", serverSelectionTimeoutMS=2000)
info = c["quant_data"]["gkg_index"].index_information()
print("yes" if "$idx_name" in info else "no")
PYEOF
)
        if [ "$exists" = "yes" ]; then
            return 0
        fi
        sleep 10
    done
}

# Stream progress until currentOp has no active Index Build operation
show_progress_until_done() {
    local label="$1"
    local log_path="$2"
    local build_pid="${3:-}"
    local idle_count=0

    echo "[$(date +%H:%M:%S)] Monitoring started: $label"
    while true; do
        result=$($VENV - <<PYEOF
from pymongo import MongoClient
import subprocess, json, re, time
client = MongoClient("$MONGO_OUTER", serverSelectionTimeoutMS=2000)
ts = time.strftime("%H:%M:%S")
try:
    ops = list(client["admin"].aggregate([
        {"\$currentOp": {"allUsers": True, "idleConnections": False}},
        {"\$match": {"msg": {"\$regex": "Index Build"}}}
    ]))
    if ops:
        op    = ops[0]
        msg   = op.get("msg", "")
        prog  = op.get("progress", {})
        done  = int(prog.get("done", 0))
        total = int(prog.get("total", 1))
        pct   = done / total * 100 if total else 0
        sr    = op.get("secs_running", 0)
        secs  = int(sr.get("low", 0) if isinstance(sr, dict) else sr)

        # Determine build phase
        if "inserting keys from external sorter" in msg:
            # Phase 3: writing B-tree
            eta  = int((100 - pct) / (pct / secs)) if pct > 0 and secs > 0 else 0
            bar  = "█" * int(40 * pct / 100) + "░" * (40 - int(40 * pct / 100))
            eta_s = f"{eta//3600}h{(eta%3600)//60}m{eta%60}s" if eta > 60 else f"{eta}s"
            print(f"PROGRESS [{ts}] [write-btree] [{bar}] {pct:5.1f}%  {done//1_000_000}/{total//1_000_000}M  elapsed={secs//60}m{secs%60}s  ETA={eta_s}")
        else:
            # Phase 1: scanning or phase 2: merging spills (progress counter is static)
            # Read recent Merging spills lines from docker logs
            try:
                log_out = subprocess.check_output(
                    ["docker", "logs", "mongo6", "--since", "30s"],
                    stderr=subprocess.STDOUT
                ).decode("utf-8", errors="replace")
                merge_lines = [l for l in log_out.splitlines() if "Merging spills" in l or "Finished merging" in l]
                if merge_lines:
                    last = merge_lines[-1]
                    m = re.search(r'"currentNumSpills":(\d+).*?"targetNumSpills":(\d+)', last)
                    fin = "Finished merging" in last
                    if fin:
                        print(f"PROGRESS [{ts}] [merge] Finished merging spills ✓ writing B-tree...")
                    elif m:
                        cur, tgt = int(m.group(1)), int(m.group(2))
                        pct_m = (1 - cur/tgt) * 100 if tgt else 0
                        bar = "█" * int(40 * pct_m / 100) + "░" * (40 - int(40 * pct_m / 100))
                        print(f"PROGRESS [{ts}] [merge] [{bar}] spills {cur}→{tgt}  elapsed={secs//60}m{secs%60}s")
                    else:
                        print(f"PROGRESS [{ts}] [merge] merging spill files...  elapsed={secs//60}m{secs%60}s")
                else:
                    # Scanning phase
                    eta  = int((100 - pct) / (pct / secs)) if pct > 0 and secs > 0 else 0
                    bar  = "█" * int(40 * pct / 100) + "░" * (40 - int(40 * pct / 100))
                    eta_s = f"{eta//3600}h{(eta%3600)//60}m{eta%60}s" if eta > 60 else f"{eta}s"
                    print(f"PROGRESS [{ts}] [scan] [{bar}] {pct:5.1f}%  {done//1_000_000}/{total//1_000_000}M  elapsed={secs//60}m{secs%60}s  ETA={eta_s}")
            except Exception as le:
                print(f"PROGRESS [{ts}] [scan/merge] {pct:.1f}%  {done//1_000_000}M  elapsed={secs//60}m{secs%60}s")
    else:
        print("IDLE")
except Exception as e:
    print(f"ERR {e}")
PYEOF
)
        if echo "$result" | grep -q "^PROGRESS"; then
            echo "$result" | grep "^PROGRESS" | sed 's/^PROGRESS //'
            idle_count=0
        elif echo "$result" | grep -q "^IDLE"; then
            # If the build process has exited, the index is truly done
            if [ -n "$build_pid" ] && ! kill -0 "$build_pid" 2>/dev/null; then
                echo "[$(date +%H:%M:%S)] Build process exited, stopping monitor"
                break
            fi
            idle_count=$((idle_count + 1))
            echo "[$(date +%H:%M:%S)] No Index Build in currentOp (waiting to start or already done) [$idle_count/3]"
            if [ $idle_count -ge 3 ]; then
                echo "[$(date +%H:%M:%S)] Build appears complete, stopping monitor"
                break
            fi
        else
            echo "[$(date +%H:%M:%S)] Query error: $result"
        fi
        sleep 15
    done

    # Print container log
    echo ""
    echo "--- container log ($log_path) ---"
    docker exec mongo6 cat "$log_path" 2>/dev/null || echo "(log file not found)"
}

# ─── Main flow ────────────────────────────────────────────────

# Increase sorter memory to reduce spill count (default 200 MB produces 8000+ spills; merge takes 50+ hours)
echo "Setting maxIndexBuildMemoryUsageMegabytes=2000 ..."
docker exec mongo6 mongosh "$MONGO_INNER" --quiet \
    --eval 'const r = db.adminCommand({setParameter:1, maxIndexBuildMemoryUsageMegabytes:2000}); print("sorter memory: " + (r.ok ? "set to 2000 MB" : "failed: " + JSON.stringify(r)))'

echo "============================================================"
echo " Current indexes:"
$VENV - <<'PYEOF'
from pymongo import MongoClient
c = MongoClient("mongodb://root:root@127.0.0.1:37018/", serverSelectionTimeoutMS=3000)
for name, info in c["quant_data"]["gkg_index"].index_information().items():
    print(f"  {name}: {info['key']}")
PYEOF
echo "============================================================"

# ─── URL index (no unique — 267 M duplicate url="http://" entries in the data) ─
URL_EXISTS=$($VENV - <<'PYEOF'
from pymongo import MongoClient
c = MongoClient("mongodb://root:root@127.0.0.1:37018/", serverSelectionTimeoutMS=2000)
print("yes" if "gkg_url" in c["quant_data"]["gkg_index"].index_information() else "no")
PYEOF
)

if [ "$URL_EXISTS" = "yes" ]; then
    echo "gkg_url already exists, skipping"
else
    echo ""
    echo "============================================================"
    echo " Starting build: gkg_url (nohup background, safe to disconnect terminal)"
    echo "============================================================"
    docker exec mongo6 mongosh \
        "mongodb://root:root@127.0.0.1:27017/?authSource=admin" --quiet \
        --eval 'db=db.getSiblingDB("quant_data"); db.gkg_index.createIndex({url:1},{name:"gkg_url"}); print("url_DONE")' \
        > "$LOG_URL" 2>&1 &
    BUILD_PID=$!
    echo "build PID: $BUILD_PID"
    sleep 3
    show_progress_until_done "gkg_url" "$LOG_URL" "$BUILD_PID"
    wait $BUILD_PID
fi

# ─── Full-text index ──────────────────────────────────────────
# Validate via a text search to confirm the index is truly usable (it appears in index_information even while building)
TEXT_EXISTS=$($VENV - <<'PYEOF'
from pymongo import MongoClient
c = MongoClient("mongodb://root:root@127.0.0.1:37018/", serverSelectionTimeoutMS=2000)
# Check if a text index build is already in progress
ops = list(c["admin"].aggregate([
    {"$currentOp": {"allUsers": True, "idleConnections": False}},
    {"$match": {"msg": {"$regex": "Index Build"}}}
]))
if ops:
    print("building")  # still building, not done
else:
    try:
        c["quant_data"]["gkg_index"].find_one({"$text": {"$search": "test"}}, {"_id": 1})
        print("yes")   # text search succeeded, index is live
    except Exception:
        print("no")    # index not usable
PYEOF
)

if [ "$TEXT_EXISTS" = "yes" ]; then
    echo "gkg_raw_text complete and usable, skipping"
elif [ "$TEXT_EXISTS" = "building" ]; then
    echo "gkg_raw_text is building, attaching monitor..."
    BUILD_PID=""
    show_progress_until_done "gkg_raw_text" "$LOG_TEXT" ""
else
    echo ""
    echo "============================================================"
    echo " Starting build: gkg_raw_text (nohup background, safe to disconnect terminal)"
    echo "============================================================"
    docker exec mongo6 mongosh \
        "mongodb://root:root@127.0.0.1:27017/?authSource=admin" --quiet \
        --eval 'db=db.getSiblingDB("quant_data"); db.gkg_index.createIndex({raw:"text"},{name:"gkg_raw_text"}); print("text_DONE")' \
        > "$LOG_TEXT" 2>&1 &
    BUILD_PID=$!
    echo "build PID: $BUILD_PID"
    sleep 3
    show_progress_until_done "gkg_raw_text" "$LOG_TEXT" "$BUILD_PID"
    wait $BUILD_PID
fi

echo ""
echo "============================================================"
echo " All done! Final indexes:"
$VENV - <<'PYEOF'
from pymongo import MongoClient
c = MongoClient("mongodb://root:root@127.0.0.1:37018/", serverSelectionTimeoutMS=3000)
for name, info in c["quant_data"]["gkg_index"].index_information().items():
    print(f"  {name}: {info['key']}")
PYEOF
echo "============================================================"
