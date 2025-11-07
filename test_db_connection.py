import mysql.connector
from pymongo import MongoClient

MYSQL_CONFIG = {
    "host": "192.168.1.26",
    "port": 13306,
    "user": "root",
    "password": "root",
    "database": "workflow"
}

MONGO_CONFIG = {
    "host": "192.168.1.26",
    "port": 27018,
    "username": "root",
    "password": "root"
}


def test_mysql():
    result = {}
    try:
        conn = mysql.connector.connect(**MYSQL_CONFIG)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM information_schema.tables;")
        count = cursor.fetchone()[0]
        result["mysql"] = f"✅ Connected ({count} tables)"
        conn.close()
    except Exception as e:
        result["mysql"] = f"❌ Connection failed: {e}"
    return result


def test_mongo():
    result = {}
    try:
        client = MongoClient(**MONGO_CONFIG)
        dbs = client.list_database_names()
        result["mongo"] = f"✅ Connected ({len(dbs)} databases)"
        client.close()
    except Exception as e:
        result["mongo"] = f"❌ Connection failed: {e}"
    return result


if __name__ == "__main__":
    res = {}
    res.update(test_mysql())
    res.update(test_mongo())
    for k, v in res.items():
        print(f"{k}: {v}")