# %%
import sys
import MySQLdb
import MySQLdb.cursors

# %%
'''
Retrieve data from MySQL
'''
conn = MySQLdb.connect(host='database-1.cfrc4kc4zmgx.ap-southeast-1.rds.amazonaws.com', db='lihkg',
                       user=sys.argv[1], passwd=sys.argv[2], charset='utf8')

try:
    with conn.cursor() as cursor:

        cursor.execute('select * from raw_data where cat_id = 5 LIMIT 0, 50')
        records = cursor.fetchall()
        for row in records:
            print(row)
        cursor.close()

finally:
    if conn:
        conn.close()