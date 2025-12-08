import time
import threading
from test_db_connection import test_mysql, test_mongo

def run_health_check():
    print("=== Quant Data System Started ===")
    print("Performing initial database connectivity test...\n")

    mysql_result = test_mysql()
    mongo_result = test_mongo()

    for name, result in {**mysql_result, **mongo_result}.items():
        print(f"{name}: {result}")

    print("\nHealth check complete.")
    print("===============================\n")

def start_health_monitor():
    """启动后台健康检查线程"""
    def loop():
        while True:
            run_health_check()
            print("Sleeping for 5 minutes before next check...\n")
            time.sleep(300)  # 每5分钟执行一次
    threading.Thread(target=loop, daemon=True).start()