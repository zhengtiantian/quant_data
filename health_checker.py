import time
import threading
from test_db_connection import test_mongo

def run_health_check():
    print("=== Quant Data System Started ===")
    print("Performing initial database connectivity test...\n")

    for name, result in test_mongo().items():
        print(f"{name}: {result}")

    print("\nHealth check complete.")
    print("===============================\n")

def start_health_monitor():
    """Start background health check thread"""
    def loop():
        while True:
            run_health_check()
            print("Sleeping for 5 minutes before next check...\n")
            time.sleep(300)  # Runs every 5 minutes
    threading.Thread(target=loop, daemon=True).start()