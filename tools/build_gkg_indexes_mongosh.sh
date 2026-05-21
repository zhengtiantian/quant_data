#!/usr/bin/env bash
# 在容器内用 nohup+mongosh 建索引，Mac 侧轮询进度
# 用法: caffeinate -i -s bash tools/build_gkg_indexes_mongosh.sh
#
# Ctrl+C 只停止进度显示，不影响容器内正在构建的索引
# 重新运行脚本可以恢复进度显示

set -euo pipefail

MONGO_INNER="mongodb://root:root@127.0.0.1:27017/?authSource=admin"
MONGO_OUTER="mongodb://root:root@127.0.0.1:37018/"
VENV=".venv311/bin/python"
LOG_URL="/tmp/build_url.log"
LOG_TEXT="/tmp/build_text.log"

# 等待某个索引出现在 index_information 中
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

# 显示进度，直到 currentOp 中没有 Index Build 任务
show_progress_until_done() {
    local label="$1"
    local log_path="$2"
    local idle_count=0

    echo "[$(date +%H:%M:%S)] 开始监控: $label"
    while true; do
        result=$($VENV - <<PYEOF
from pymongo import MongoClient
import time
client = MongoClient("$MONGO_OUTER", serverSelectionTimeoutMS=2000)
ts = time.strftime("%H:%M:%S")
try:
    ops = list(client["admin"].aggregate([
        {"\$currentOp": {"allUsers": True, "idleConnections": False}},
        {"\$match": {"msg": {"\$regex": "Index Build"}}}
    ]))
    for op in ops:
        prog  = op.get("progress", {})
        done  = int(prog.get("done", 0))
        total = int(prog.get("total", 1))
        pct   = done / total * 100 if total else 0
        sr    = op.get("secs_running", 0)
        secs  = int(sr.get("low", 0) if isinstance(sr, dict) else sr)
        eta   = int((100 - pct) / (pct / secs)) if pct > 0 and secs > 0 else 0
        bar   = "█" * int(40 * pct / 100) + "░" * (40 - int(40 * pct / 100))
        eta_s = f"{eta//3600}h{(eta%3600)//60}m{eta%60}s" if eta > 60 else f"{eta}s"
        print(f"PROGRESS [{ts}] [{bar}] {pct:5.1f}%  {done//1_000_000}/{total//1_000_000}M  elapsed={secs//60}m{secs%60}s  ETA={eta_s}")
    if not ops:
        print("IDLE")
except Exception as e:
    print(f"ERR {e}")
PYEOF
)
        if echo "$result" | grep -q "^PROGRESS"; then
            echo "$result" | grep "^PROGRESS" | sed 's/^PROGRESS //'
            idle_count=0
        elif echo "$result" | grep -q "^IDLE"; then
            idle_count=$((idle_count + 1))
            echo "[$(date +%H:%M:%S)] currentOp 无 Index Build（等待启动或已完成）[$idle_count/3]"
            if [ $idle_count -ge 3 ]; then
                echo "[$(date +%H:%M:%S)] 判断构建已完成，退出监控"
                break
            fi
        else
            echo "[$(date +%H:%M:%S)] 查询失败: $result"
        fi
        sleep 15
    done

    # 打印容器内日志
    echo ""
    echo "--- 容器内日志 ($log_path) ---"
    docker exec mongo6 cat "$log_path" 2>/dev/null || echo "(日志文件不存在)"
}

# ─── 主流程 ───────────────────────────────────────────────────

# 调大 sorter 内存，减少 spill 文件数量（默认 200MB 会产生 8000+ spill，merge 需 50+ 小时）
echo "设置 maxIndexBuildMemoryUsageMegabytes=3000 ..."
docker exec mongo6 mongosh "$MONGO_INNER" --quiet \
    --eval 'const r = db.adminCommand({setParameter:1, maxIndexBuildMemoryUsageMegabytes:3000}); print("sorter 内存: " + (r.ok ? "已设为 3000MB" : "设置失败: " + JSON.stringify(r)))'

echo "============================================================"
echo " 当前索引:"
$VENV - <<'PYEOF'
from pymongo import MongoClient
c = MongoClient("mongodb://root:root@127.0.0.1:37018/", serverSelectionTimeoutMS=3000)
for name, info in c["quant_data"]["gkg_index"].index_information().items():
    print(f"  {name}: {info['key']}")
PYEOF
echo "============================================================"

# ─── URL 索引（不带 unique，数据中有 2.67亿条重复 url="http://"）─────
URL_EXISTS=$($VENV - <<'PYEOF'
from pymongo import MongoClient
c = MongoClient("mongodb://root:root@127.0.0.1:37018/", serverSelectionTimeoutMS=2000)
print("yes" if "gkg_url" in c["quant_data"]["gkg_index"].index_information() else "no")
PYEOF
)

if [ "$URL_EXISTS" = "yes" ]; then
    echo "gkg_url 已存在，跳过"
else
    echo ""
    echo "============================================================"
    echo " 启动构建: gkg_url（nohup 后台，断开终端不影响）"
    echo "============================================================"
    docker exec mongo6 bash -c \
        'nohup mongosh "mongodb://root:root@127.0.0.1:27017/?authSource=admin" --quiet \
         --eval "db=db.getSiblingDB(\"quant_data\"); db.gkg_index.createIndex({url:1},{name:\"gkg_url\"}); print(\"url_DONE\")" \
         > '"$LOG_URL"' 2>&1 &
         echo "mongosh PID: $!"'
    sleep 3
    show_progress_until_done "gkg_url" "$LOG_URL"
fi

# ─── Text 全文索引 ─────────────────────────────────────────────
TEXT_EXISTS=$($VENV - <<'PYEOF'
from pymongo import MongoClient
c = MongoClient("mongodb://root:root@127.0.0.1:37018/", serverSelectionTimeoutMS=2000)
print("yes" if "gkg_raw_text" in c["quant_data"]["gkg_index"].index_information() else "no")
PYEOF
)

if [ "$TEXT_EXISTS" = "yes" ]; then
    echo "gkg_raw_text 已存在，跳过"
else
    echo ""
    echo "============================================================"
    echo " 启动构建: gkg_raw_text（nohup 后台，断开终端不影响）"
    echo "============================================================"
    docker exec mongo6 bash -c \
        'nohup mongosh "mongodb://root:root@127.0.0.1:27017/?authSource=admin" --quiet \
         --eval "db=db.getSiblingDB(\"quant_data\"); db.gkg_index.createIndex({raw:\"text\"},{name:\"gkg_raw_text\"}); print(\"text_DONE\")" \
         > '"$LOG_TEXT"' 2>&1 &
         echo "mongosh PID: $!"'
    sleep 3
    show_progress_until_done "gkg_raw_text" "$LOG_TEXT"
fi

echo ""
echo "============================================================"
echo " 全部完成！最终索引:"
$VENV - <<'PYEOF'
from pymongo import MongoClient
c = MongoClient("mongodb://root:root@127.0.0.1:37018/", serverSelectionTimeoutMS=3000)
for name, info in c["quant_data"]["gkg_index"].index_information().items():
    print(f"  {name}: {info['key']}")
PYEOF
echo "============================================================"
