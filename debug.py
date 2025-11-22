
import pymysql
try:
    pymysql.connect(
        host="localhost",
        user="ai",
        password="8662533",
        database="eurusd"
    )
    print("CONNECTED")
except Exception as e:
    print("FAILED:", e)
EOF
