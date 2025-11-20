import time
from test_db_connection import test_mysql, test_mongo


def run_health_check():
    print("=== Quant Data System Started 1===")
    print("Performing initial database connectivity test...\n")

    mysql_result = test_mysql()
    mongo_result = test_mongo()

    for name, result in {**mysql_result, **mongo_result}.items():
        print(f"{name}: {result}")

    print("\nHealth check complete.")
    print("===============================")


if __name__ == "__main__":

    while True:
        run_health_check()
        print("Sleeping for 5 minutes before next check...\n")
        time.sleep(300)  # 每 5 分钟执行一次