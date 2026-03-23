# Test nhanh, chạy riêng trong terminal
import psycopg2
conn = psycopg2.connect(host="postgres", dbname="optimize", user="postgres", password="password")
print(conn.status)