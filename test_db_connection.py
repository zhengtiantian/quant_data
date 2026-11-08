"""Container health check: MongoDB answers a ping.

MySQL was retired on 2026-10-31 (Keycloak and the GDELT queue moved to PostgreSQL), and
mysql-connector was never in requirements.txt, so the old version failed on import. Exits 0 when
healthy and 1 otherwise, so the Dockerfile HEALTHCHECK can use the exit status directly.
"""
import os
import sys

from pymongo import MongoClient

MONGO_URI = os.getenv("MONGO_URI", "mongodb://root:root@mongo6:27017/")


def test_mongo(timeout_ms: int = 5000) -> dict:
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=timeout_ms)
        try:
            client.admin.command("ping")
            count = len(client.list_database_names())
        finally:
            client.close()
        return {"mongo": f"✅ Connected ({count} databases)"}
    except Exception as e:  # noqa: BLE001 - any failure means unhealthy
        return {"mongo": f"❌ Connection failed: {e}"}


if __name__ == "__main__":
    result = test_mongo()
    for name, status in result.items():
        print(f"{name}: {status}")
    sys.exit(0 if all(s.startswith("✅") for s in result.values()) else 1)
